// SPDX-FileCopyrightText: 2026 Kenta IDA <fuga@fugafuga.org>
// SPDX-License-Identifier: BSL-1.0

#include "render_task.hpp"

#include <cstdio>
#include <string>

#include <esp_log.h>
#include <esp_psram.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "aora_face.hpp"
#include "avatar/avatar.hpp"
#include "avatar/canvas.hpp"
#include "avatar/canvas_m5gfx.hpp"
#include "face_config.hpp"
#include "screens.hpp"

namespace stackchan::app {

namespace {

constexpr const char* kTag = "render";
// Frame pacing: the target period has the frame's own draw time deducted, so
// the face actually runs near the target instead of target-plus-draw-time.
// Early-exit paths (screen off / overlay) keep the relaxed cadence.
constexpr TickType_t kPeriodTicks = pdMS_TO_TICKS(33);
constexpr std::uint32_t kTargetFrameMs = 17; // ~60 fps when drawing fits

using avatar::RichCanvas;

// One-touch mute badge: a struck-through speaker glyph shown whenever
// speaker_muted is set, so the user can tell at a glance why the device is
// silent. Drawn in the top-left corner — on touch boards the same corner is
// the tap zone (device_ui::handle_tap) that toggles the flag, so the badge
// doubles as the "tap here to unmute" affordance.
void draw_mute_badge(RichCanvas& canvas, int x, int y) {
    constexpr int w = 34, h = 16;
    const std::uint16_t white = canvas.color565(235, 235, 235);
    const std::uint16_t black = canvas.color565(0, 0, 0);
    const std::uint16_t red = canvas.color565(230, 110, 110);

    canvas.begin_group(x - 1, y - 1, w + 2, h + 2);
    canvas.fillRoundRect(x - 1, y - 1, w + 2, h + 2, 3, black);
    canvas.drawRoundRect(x - 1, y - 1, w + 2, h + 2, 3, red);
    // Speaker glyph: driver box + cone.
    canvas.fillRect(x + 5, y + 5, 4, 6, white);
    canvas.fillTriangle(x + 9, y + 8, x + 16, y + 1, x + 16, y + 15, white);
    // Diagonal red bar across the glyph (two triangles — the Canvas
    // abstraction has no thick-line primitive).
    canvas.fillTriangle(x + 20, y + 1, x + 24, y + 1, x + 32, y + 15, red);
    canvas.fillTriangle(x + 20, y + 1, x + 28, y + 15, x + 32, y + 15, red);
    canvas.end_group();
}

void render_task_entry(void* arg) {
    auto& args = *static_cast<RenderTaskArgs*>(arg);
    M5GFX& display = *args.display;
    SharedState* state = args.state;

    // The drawing strategy (= how the system reaches the panel) is owned here
    // (main), not by the avatar / on-device UI — they render through the
    // abstract Canvas. Chosen at runtime by PSRAM availability:
    //   - PSRAM present: BufferedCanvas — one full-screen sprite, pushed once.
    //   - PSRAM absent:  DirectCanvas   — draw to the panel + small scratch.
    // Both objects are cheap to construct (no buffer until used), so we hold
    // both and bind the base reference to the chosen one.
    // Canvas dims follow the display at runtime (CoreS3 = 320x240,
    // AtomS3R = 128x128) so the avatar / device-UI / balloon all adapt
    // without per-board ifdefs.
    const std::int32_t canvas_w = display.width();
    const std::int32_t canvas_h = display.height();
    avatar::BufferedCanvas buffered{display, args.circular_display};
    avatar::DirectCanvas direct{display, args.circular_display};
    avatar::RichCanvas* cv = nullptr;
    // PSRAM presence drives the buffered (full framebuffer) vs direct
    // (partial-update) canvas choice. When SPIRAM is disabled at compile
    // time (AtomS3 slim profile) esp_psram_get_size is removed from the
    // build, so guard the call out and force the direct path.
#if CONFIG_SPIRAM
    // The direct (unbuffered) path was measured at 10-16 fps but flickers constantly without double buffering,
    // so the buffered path is used. Faster frame structures (internal-RAM canvas / dirty rectangles) are not done yet.
    const bool has_psram = esp_psram_get_size() > 0;
#else
    const bool has_psram = false;
#endif
    if (has_psram && buffered.begin(canvas_w, canvas_h)) {
        cv = &buffered;
        ESP_LOGI(kTag, "PSRAM detected: buffered full-screen framebuffer (%dx%d)", static_cast<int>(canvas_w),
                 static_cast<int>(canvas_h));
    } else {
        direct.begin();
        cv = &direct;
        ESP_LOGI(kTag, "no PSRAM framebuffer: direct + partial-buffer rendering (%dx%d)", static_cast<int>(canvas_w),
                 static_cast<int>(canvas_h));
    }
    RichCanvas& canvas = *cv;

    avatar::Avatar avatar;
    // The embedded Grok DSL is the only face. Clearing an uploaded bytecode
    // override reverts the VM to that firmware default.

    std::int32_t last_mood = -1;
    std::int32_t last_voice = -1;
    std::uint32_t last_overlay_cmd = 0;
    std::uint32_t last_activity_seq = 0;
    std::uint32_t last_balloon_version = 0;
    std::uint32_t last_face_config_version = 0;
    std::uint32_t last_face_bytecode_version = 0;
    std::string balloon_scratch;
    bool balloon_pending = false;
    bool ui_was_active = false;
    bool last_muted = false;

    for (;;) {
        const std::uint32_t now_ms = static_cast<std::uint32_t>(esp_timer_get_time() / 1000);

        // Camera session: the sprite compose + panel push is the dominant
        // PSRAM bandwidth consumer, and the camera's EDMA writes its frame
        // into a PSRAM fb — running both concurrently tears the captured
        // image (bands of corrupt pixels; confirmed on hardware at RGB565's
        // 2 B/px rate). Freeze the face for the ~1.5 s session.
        if (state->i2c_quiesce.load(std::memory_order_acquire)) {
            ui_was_active = true; // full repaint when the session ends
            vTaskDelay(kPeriodTicks);
            continue;
        }

        // A full-screen overlay (SoftAP QR screen > per-board settings UI —
        // priority lives in screens::init) takes the panel over from the
        // avatar. Canvas-based overlays render lazily and report whether
        // they repainted (present only then); the AP screen paints the
        // panel directly (M5GFX::qrcode isn't in the Canvas abstraction)
        // and returns false so the stale canvas is never pushed over it.
        if (screens::overlay_active()) {
            if (screens::draw_overlay(canvas)) {
                canvas.end_frame();
            }
            ui_was_active = true; // force avatar full-repaint on exit
            vTaskDelay(kPeriodTicks);
            continue;
        }
        if (ui_was_active) {
            // Returning to the avatar — force a full repaint so the direct
            // strategy clears the whole panel (UI content) before redrawing.
            ui_was_active = false;
            avatar.request_full_repaint();
        }

        // Live face-tuning updates (BLE settings UI / boot-time NVS restore).
        // Parsing the JSON here keeps it off the BLE host task's small stack.
        const std::uint32_t face_config_version = args.state->face_config_version();
        if (face_config_version != last_face_config_version) {
            avatar.set_face_tuning(parse_face_tuning(args.state->snapshot_face_config()));
            last_face_config_version = face_config_version;
        }

        // Live face DSL bytecode swap. The HTTP / BLE upload sinks push the
        // raw .avbc into SharedState here (off the host task), and we apply it
        // — empty payload means "revert to firmware default".
        const std::uint32_t face_bc_version = args.state->face_bytecode_version();
        if (face_bc_version != last_face_bytecode_version) {
            auto bc = args.state->snapshot_face_bytecode();
            if (bc.empty()) {
                avatar.reset_face_bytecode();
            } else {
                avatar.load_face_bytecode(bc);
            }
            avatar.request_full_repaint();
            last_face_bytecode_version = face_bc_version;
        }

        // Mute badge appears/disappears with the flag; the direct (partial-
        // update) strategy never repaints untouched pixels, so force a full
        // repaint on each edge or the stale badge lingers after unmute.
        const bool muted_now = state->speaker.muted.load(std::memory_order_relaxed);
        if (muted_now != last_muted) {
            last_muted = muted_now;
            avatar.request_full_repaint();
        }

        // Mood falls back automatically (the resting state is the idle face): the mood written by
        // BtnB / shaking / MCP / the LLM persists, so a leftover test value would hold the face forever.
        // After 45 s without a new write it falls back to Neutral; during a conversation the LLM keeps
        // changing the mood, which refreshes the timestamp. Idle decay (bored after 2 min, sleepy after
        // 5 min) takes over after the fallback as usual.
        std::int32_t mood = args.state->face.expression.load(std::memory_order_relaxed);
        static std::uint32_t mood_set_ms = 0;
        constexpr std::uint32_t kMoodAutoClearMs = 45'000;
        if (mood != last_mood) {
            mood_set_ms = now_ms;
        } else if (mood != 0 && now_ms - mood_set_ms >= kMoodAutoClearMs) {
            args.state->face.expression.store(0, std::memory_order_relaxed);
            mood = 0;
        }
        if (mood != last_mood) {
            avatar.set_expression(static_cast<avatar::Expression>(mood));
            last_mood = mood;
        }
        const std::int32_t voice =
            static_cast<std::int32_t>(args.state->conv.voice_state.load(std::memory_order_relaxed));
        if (voice != last_voice) {
            avatar.set_voice_state(static_cast<avatar::VoiceState>(voice));
            last_voice = voice;
        }
        // One atomic word per overlay command: a concurrent writer can never
        // hand us the expression of one command and the hold of another.
        const std::uint32_t overlay_cmd = args.state->face.overlay_command.load(std::memory_order_acquire);
        if (overlay_cmd != last_overlay_cmd) {
            last_overlay_cmd = overlay_cmd;
            if ((overlay_cmd & app::SharedState::kOverlayValidBit) == 0) {
                avatar.clear_overlay();
            } else {
                const auto expr = static_cast<avatar::Expression>((overlay_cmd >> 15) & 0xFF);
                const auto hold = overlay_cmd & 0x7FFFu;
                avatar.set_overlay(expr, hold);
            }
        }
        const std::uint32_t activity_seq = args.state->face.activity_seq.load(std::memory_order_relaxed);
        if (activity_seq != last_activity_seq) {
            last_activity_seq = activity_seq;
            avatar.note_activity();
        }
        const float mouth_open = args.state->face.mouth_open.load(std::memory_order_relaxed);
        avatar.set_mouth_open(mouth_open);
        avatar.set_voice_level(args.state->face.voice_level.load(std::memory_order_relaxed));
        avatar.set_gaze(args.state->face.gaze_target_h.load(std::memory_order_relaxed),
                        args.state->face.gaze_target_v.load(std::memory_order_relaxed));

        const std::uint32_t balloon_version = args.state->balloon_version();
        if (balloon_version != last_balloon_version) {
            if (args.state->balloon_visible()) {
                std::uint32_t hold_ms = 0;
                args.state->snapshot_balloon(balloon_scratch, hold_ms);
                avatar.set_balloon_text(balloon_scratch, hold_ms);
                balloon_pending = true;
            } else {
                avatar.clear_balloon();
                balloon_pending = false;
            }
            last_balloon_version = balloon_version;
        }

        // Grok (Avatar VM) is the only face.
        {
            // aora eye rings: build this frame's eye shape from the expression weights after the previous
            // tick (one frame late, not visible); the VM reads them through the Var::RingBase range.
            static float ring_buf[aora::kOutFloats];
            aora::compose(avatar.draw_context(), now_ms, ring_buf);
            avatar.set_eye_ring_buffer(ring_buf);
            // avatar.tick() opens the frame (begin_frame) and draws the face;
            // overlays compose into the same frame; end_frame() presents.
            avatar.tick(now_ms, canvas);
        }

        // The battery gauge is not drawn on the face: the value is still
        // available in SharedState and on the settings page. The badge layout keeps its place.
        const bool gauge_shown = false;

        // Mute badge, below the battery gauge when both are up. Round
        // panels (StopWatch) inset it to the inscribed square so it stays
        // on the visible circle.
        if (state->speaker.muted.load(std::memory_order_relaxed)) {
            const int inset = args.circular_display ? static_cast<int>(canvas_w * (1.0f - 0.70710678f) * 0.5f) + 4 : 6;
            draw_mute_badge(canvas, inset, inset + (gauge_shown ? 22 : 0));
        }

        canvas.end_frame();

        if (balloon_pending && avatar.is_balloon_done()) {
            balloon_pending = false;
            args.state->notify_balloon_complete();
        }

        // Deduct this frame's draw time from the target period. A full-screen
        // draw (~140 ms) overruns the budget every frame, and a 1-tick sleep
        // starves IDLE1 when another core-1 task is also busy: task_wdt then
        // barks every 5 s. Keep a 20 ms floor so IDLE always feeds the dog.
        {
            constexpr TickType_t kRestFloor = pdMS_TO_TICKS(20);
            const std::uint32_t frame_end = static_cast<std::uint32_t>(esp_timer_get_time() / 1000);
            const std::uint32_t spent = frame_end - now_ms;
            TickType_t rest = spent >= kTargetFrameMs ? kRestFloor : pdMS_TO_TICKS(kTargetFrameMs - spent);
            if (rest < kRestFloor) {
                rest = kRestFloor;
            }
            vTaskDelay(rest);

            // Coarse FPS log every ~5 s so real-device rate is visible.
            static std::uint32_t fps_frames = 0;
            static std::uint32_t fps_window_start = 0;
            ++fps_frames;
            if (fps_window_start == 0) {
                fps_window_start = frame_end;
            } else if (frame_end - fps_window_start >= 5000) {
                ESP_LOGI(kTag, "render ~%u fps (draw %u ms this frame)",
                         static_cast<unsigned>(fps_frames * 1000u / (frame_end - fps_window_start)),
                         static_cast<unsigned>(spent));
                fps_frames = 0;
                fps_window_start = frame_end;
            }
        }
    }
}

} // namespace

void start_render_task(RenderTaskArgs& args) {
    xTaskCreatePinnedToCore(render_task_entry, "render", 8192, &args, 5, nullptr, 1);
}

} // namespace stackchan::app
