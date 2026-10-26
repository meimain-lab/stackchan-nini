// SPDX-FileCopyrightText: 2026 Kenta IDA <fuga@fugafuga.org>
// SPDX-License-Identifier: BSL-1.0

#include "conversation_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include <M5Unified.h>
#include <cJSON.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/idf_additions.h>
#include <freertos/queue.h>
#include <freertos/task.h>

#include "avatar/expression.hpp"
#include "avatar/expression_names.hpp"
#include "board/si12t_touch.hpp"
#include "conversation/gemini_live_client.hpp"
#include "conversation/head_command.hpp"
#include "conversation/metrics.hpp"
#include "conversation/openai_realtime_client.hpp"
#include "conversation/xiaozhi_client.hpp"
#include "led_runtime.hpp"
#include "utf8.hpp"
#include "wifi_sta.hpp"

#include <jtts/jtts.hpp>

namespace stackchan::app {

namespace {

namespace conv = stackchan::conversation;

constexpr const char* kTag = "conv-task";

// Audio rates depend on the selected provider:
//   OpenAI Realtime: µ-law @ 8 kHz both directions (encoded inside the
//                    client, this task just deals in PCM16).
//   Gemini Live:     PCM16 @ 16 kHz uplink, PCM16 @ 24 kHz downlink.
// The values used here are PCM16 sample rates the client expects/produces;
// any companding happens inside the ConversationService impl.
constexpr std::size_t kMaxMicChunkSamples = 640; // worst case: 40 ms @ 16 kHz
constexpr std::uint32_t kEnvelopeStepMs = 16;

// Playback ring: M5.Speaker.playRaw references the buffer (no copy) and its
// resampler reads it sample-by-sample. If that buffer is in PSRAM it contends
// with the render task's 30 fps sprite traffic and the I2S DMA underruns
// (choppy playback). So the reply is accumulated in PSRAM but played back in
// segments copied into this small internal-RAM ring — the speaker only ever
// reads from fast SRAM. 3 buffers: M5.Speaker holds 2, we always have 1 free.
// The sizes live in conversation_task.hpp so app_main can pre-allocate the
// ring at boot — see ConversationTaskArgs::seg_buf.
constexpr std::size_t kSegmentSamples = kConversationSegmentSamples;
constexpr std::size_t kSegmentBuffers = kConversationSegmentBuffers;
constexpr int kSpeakerChannel = 0;

// Streaming playback: start speaking once this many ms of reply audio has
// been buffered, rather than waiting for the whole reply. Jitter margin
// against network hiccups. Scaled to actual speaker_sample_rate_ at use.
constexpr std::uint32_t kJitterBufferMs = 300;

// Mic / speaker I2S handoff settle time (matches the existing audio code).
constexpr TickType_t kI2sSettle = pdMS_TO_TICKS(20);

// Health-check cadence while Thinking: an alive session keeps waiting for
// the reply; only a dead session recovers. This is a check interval, not a
// face deadline.
constexpr std::uint32_t kThinkingTimeoutMs = 60000;

// How long the thinking face may stay. Background noise that the cloud VAD takes for speech can move the
// state to Thinking with no reply ever coming, which would leave the thinking face up forever. Real
// replies almost always start within 12 s; after that the face goes back to idle while the state machine
// keeps waiting (a late reply still moves to Speaking as usual).
constexpr std::uint32_t kThinkingFaceMs = 12000;

// Recover from a stuck "Speaking" (assistant playback drained but the
// generationComplete / interrupted / turnComplete event never arrived,
// so finish_speaking() can't fire on its own). See GitHub issue #2.
// This is a stall timeout, not a cap on reply length: it only counts while
// every received sample has been queued, the speaker is idle and no new
// audio has arrived, so long replies play to the end.
constexpr std::uint32_t kSpeakingStallTimeoutMs = 20000;
// Once this much reply audio has been handed to the speaker, drop it from
// the front of assistant_pcm_ so a long reply doesn't keep growing the
// PSRAM buffer (60 s at 24 kHz is ~3 MB).
constexpr std::uint32_t kPcmTrimAfterMs = 2000;
constexpr std::uint32_t kReconnectBackoffBaseMs = 500;
constexpr std::uint32_t kReconnectBackoffCapMs = 30u * 1000u;

// Local half-duplex state machine. Distinct from conv::ConversationState
// (which tracks the protocol) because we hold "Speaking" until the speaker
// has physically drained, independent of when response.done arrives.
enum class Local : std::uint8_t {
    Init,       // waiting for the session to reach Listening
    Listening,  // mic streaming up
    Thinking,   // speech_stopped seen; accumulating the reply
    Speaking,   // playing the reply through the speaker
    Yielded,    // I2S handed off to BLE audio streamer (mic + speaker ended)
};

// Built-in persona, used when the device setting `system-prompt` is empty.
// Kept language-neutral: the robot answers in the language the user speaks.
const char* kInstructions =
    "You are Groki, a small desktop robot running on an M5Stack CoreS3. "
    "You are friendly and cheerful, and you keep replies short. "
    "Reply in the language the user speaks. "
    "If you are asked which AI model you run on, answer truthfully. "
    "You can change your facial expression and where your head points: use the set_expression and "
    "set_head_pose tools to match the mood. "
    "The speak_katakoto tool makes short robot-style Japanese kana sounds. Use it only for sound effects, "
    "imitation and short noises; after calling it you do not need to repeat the same thing in normal speech.";

conv::ToolDefinition make_set_expression_tool()
{
    return conv::ToolDefinition{
        .name = "set_expression",
        .description = "Change Groki's facial expression. Use it to show emotion.",
        .parameters_json = R"({"type":"object","properties":{"expression":{"type":"string",)"
                           R"("enum":["neutral","idle","happy","sad","angry","doubt","sleepy",)"
                           R"("listening","thinking","excited","curious","confused",)"
                           R"("surprised","dizzy","affection","bored"]}},)"
                           R"("required":["expression"]})",
    };
}

conv::ToolDefinition make_set_head_pose_tool()
{
    return conv::ToolDefinition{
        .name = "set_head_pose",
        .description = "Turn Groki's head. yaw is left/right (-40 to 40 degrees), pitch is up/down (-10 to 25 degrees).",
        .parameters_json =
            R"({"type":"object","properties":{)"
            R"("yaw_deg":{"type":"number"},"pitch_deg":{"type":"number"}},"required":["yaw_deg","pitch_deg"]})",
    };
}

conv::ToolDefinition make_speak_katakoto_tool()
{
    return conv::ToolDefinition{
        .name = "speak_katakoto",
        .description =
            "Make a short robot-style sound in Japanese kana. "
            "kana may contain only hiragana, katakana, the long-vowel mark \"ー\", the small \"っ\" and spaces; "
            "no kanji. Examples: \"ぴこーん\" / \"こんにちわー\" / \"がんばるぞー\".",
        .parameters_json =
            R"({"type":"object","properties":{)"
            R"("kana":{"type":"string","maxLength":48}},"required":["kana"]})",
    };
}

std::optional<avatar::Expression> parse_expression(const char* name) {
    return avatar::expression_from_name(name);
}

// Map XiaoZhi's affective `llm.emotion` vocabulary onto the expanded
// expression set. Direct names (thinking/confused/surprised/…) hit
// parse_expression first; the rest collapse onto the closest face.
// Unknown names yield nullopt and are ignored by the caller.
std::optional<avatar::Expression> parse_emotion(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }
    // Direct hits on our own vocabulary first.
    if (auto e = parse_expression(name)) {
        return e;
    }
    if (std::strcmp(name, "laughing") == 0 || std::strcmp(name, "funny") == 0) {
        return avatar::Expression::Excited;
    }
    if (std::strcmp(name, "loving") == 0 || std::strcmp(name, "delicious") == 0 || std::strcmp(name, "kissy") == 0 ||
        std::strcmp(name, "winking") == 0 || std::strcmp(name, "silly") == 0) {
        return avatar::Expression::Happy;
    }
    if (std::strcmp(name, "crying") == 0) {
        return avatar::Expression::Sad;
    }
    if (std::strcmp(name, "shocked") == 0) {
        return avatar::Expression::Surprised;
    }
    if (std::strcmp(name, "embarrassed") == 0) {
        return avatar::Expression::Doubt;
    }
    if (std::strcmp(name, "relaxed") == 0 || std::strcmp(name, "cool") == 0 || std::strcmp(name, "confident") == 0) {
        return avatar::Expression::Neutral;
    }
    return std::nullopt;
}

std::uint32_t now_ms() {
    return static_cast<std::uint32_t>(esp_timer_get_time() / 1000);
}

std::int64_t now_us() {
    return esp_timer_get_time();
}

std::uint32_t reconnect_backoff_ms(unsigned consecutive_failures) {
    if (consecutive_failures == 0) {
        return 0;
    }
    std::uint32_t backoff = kReconnectBackoffBaseMs;
    for (unsigned i = 1; i < consecutive_failures; ++i) {
        if (backoff >= kReconnectBackoffCapMs / 2) {
            return kReconnectBackoffCapMs;
        }
        backoff *= 2;
    }
    return std::min(backoff, kReconnectBackoffCapMs);
}

// Owns the conversation; one instance per task.
class Coordinator {
public:
    Coordinator(SharedState& state, const char* api_key, config::Provider provider, board::Si12tTouch* touch,
                const char* xiaozhi_url, const char* xiaozhi_token, const char* system_prompt,
                const char* extra_headers, const ServoLimits& limits,
                const std::array<std::int16_t*, kSegmentBuffers>& seg_buf)
        : state_{state}, api_key_{api_key != nullptr ? api_key : ""}, provider_{provider}, touch_{touch},
          xiaozhi_url_{xiaozhi_url != nullptr ? xiaozhi_url : ""},
          xiaozhi_token_{xiaozhi_token != nullptr ? xiaozhi_token : ""},
          system_prompt_{system_prompt != nullptr ? system_prompt : ""},
          extra_headers_{extra_headers != nullptr ? extra_headers : ""}, limits_{limits}, seg_buf_{seg_buf} {
        // Per-provider audio rates. The OpenAI client further compands its
        // 8 kHz PCM16 into µ-law on the wire; Gemini sends raw PCM16; XiaoZhi
        // streams Opus (16 kHz mono up, server-rate down, resampled to 24 kHz).
        if (provider_ == config::Provider::Gemini || provider_ == config::Provider::XiaoZhi) {
            mic_sample_rate_ = 16000;
            speaker_sample_rate_ = 24000;
        } else {
            mic_sample_rate_ = 8000;
            speaker_sample_rate_ = 8000;
        }
        mic_chunk_samples_ = mic_sample_rate_ * 40 / 1000;        // 40 ms per chunk
        jitter_buffer_samples_ = speaker_sample_rate_ * kJitterBufferMs / 1000u;
    }

    void run()
    {
        // XiaoZhi is keyed by its server URL rather than an API key; the other
        // providers need a non-empty api_key.
        const bool disabled = provider_ == config::Provider::XiaoZhi ? xiaozhi_url_.empty()
                                                                     : api_key_.empty();
        if (disabled) {
            ESP_LOGW(kTag, "credentials empty for selected provider — conversation disabled");
            vTaskDelete(nullptr);
            return;
        }

        ESP_LOGI(kTag, "waiting for Wi-Fi...");
        state_.conv.status.store(ConvStatus::WaitingWifi, std::memory_order_relaxed);
        while (!wifi_is_connected()) {
            vTaskDelay(pdMS_TO_TICKS(500));
        }
        ESP_LOGI(kTag, "Wi-Fi up, starting %s conversation", provider_name());

        event_queue_ = xQueueCreate(32, sizeof(conv::ConversationEvent*));
        mic_buf_[0].resize(mic_chunk_samples_);
        mic_buf_[1].resize(mic_chunk_samples_);

        // Segment ring must live in internal RAM (see kSegmentSamples comment).
        // Allocation now happens in app_main during the boot-time
        // low-fragmentation window — see ConversationTaskArgs::seg_buf. If
        // app_main couldn't reserve them (extremely tight internal RAM at
        // boot), we get nullptr entries here and disable the task instead
        // of trying to alloc late and failing harder.
        for (auto* buf : seg_buf_) {
            if (buf == nullptr) {
                ESP_LOGE(kTag,
                         "seg_buf not provided by app_main — conversation disabled "
                         "(boot-time internal-RAM reservation failed)");
                state_.conv.status.store(ConvStatus::Error, std::memory_order_relaxed);
                vTaskDelete(nullptr);
                return;
            }
        }

        if (provider_ == config::Provider::Gemini) {
            client_ = std::make_unique<conv::GeminiLiveClient>(api_key_);
        } else if (provider_ == config::Provider::XiaoZhi) {
            client_ = std::make_unique<conv::XiaoZhiClient>(xiaozhi_url_, xiaozhi_token_);
        } else {
            client_ = std::make_unique<conv::OpenAiRealtimeClient>(api_key_);
        }
        client_->set_event_callback([this](const conv::ConversationEvent& ev) { enqueue_event(ev); });

        (void)connect_with_retries("initial connect", /*allow_audio_abort=*/false);

        for (;;) {
            // Full conversation shutdown for BLE audio streaming. Yielding
            // just the mic isn't enough — the live WebSocket + mbedtls
            // session keeps Wi-Fi pumping and squeezes internal RAM,
            // which starves the AAC decoder (observed: error 30 only
            // after Wi-Fi associates). When audio_stream_active fires we
            // stop the client entirely so internal RAM rebounds; the
            // streamer publishes conversation_yielded_i2s once we're
            // fully torn down. On clear we reconnect from scratch.
            if (state_.audio_stream_active.load(std::memory_order_acquire)) {
                if (local_ != Local::Yielded) {
                    ESP_LOGI(kTag, "yielding to BLE audio stream — stopping conversation");
                    M5.Mic.end();
                    M5.Speaker.end();
                    state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
                    client_->stop();
                    vTaskDelay(pdMS_TO_TICKS(500));
                    flush_events();
                    assistant_pcm_.clear();
                    assistant_text_.clear();
                    set_local(Local::Yielded);
                    state_.conv.active.store(false, std::memory_order_relaxed);
                    state_.conv.idle.store(false, std::memory_order_relaxed);
                    state_.conv.yielded_i2s.store(true, std::memory_order_release);
                }
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
            if (local_ == Local::Yielded) {
                ESP_LOGI(kTag, "BLE audio done — restarting conversation");
                state_.conv.yielded_i2s.store(false, std::memory_order_release);
                if (!connect_with_retries("reconnect after BLE audio",
                                          /*allow_audio_abort=*/true)) {
                    continue;
                }
                // connect() leaves us in Local::Init waiting for the
                // session-ready event; outer loop will drain_events()
                // and the existing handler progresses us to Listening.
            }
            drain_events();
            service_state();
        }
    }

private:
    // ---- event bridge: WS task -> this task --------------------------------

    void enqueue_event(const conv::ConversationEvent& ev)
    {
        auto* copy = new conv::ConversationEvent(ev);
        if (xQueueSend(event_queue_, &copy, 0) != pdTRUE) {
            ESP_LOGW(kTag, "event queue full, dropping %d", static_cast<int>(ev.type));
            delete copy;
        }
    }

    void drain_events()
    {
        conv::ConversationEvent* ev = nullptr;
        while (xQueueReceive(event_queue_, &ev, 0) == pdTRUE) {
            handle_event(*ev);
            delete ev;
        }
    }

    // Discard whatever's in the event queue without handling. Used by the
    // recovery path to drop stale events emitted by the WebSocket client we
    // just shut down — otherwise the freshly-connected new client gets
    // immediately knocked over by a left-over DISCONNECTED from the old one.
    void flush_events()
    {
        conv::ConversationEvent* ev = nullptr;
        std::size_t dropped = 0;
        while (xQueueReceive(event_queue_, &ev, 0) == pdTRUE) {
            delete ev;
            ++dropped;
        }
        if (dropped > 0) {
            ESP_LOGD(kTag, "flushed %u stale events", static_cast<unsigned>(dropped));
        }
    }

    // ---- session lifecycle -------------------------------------------------

    bool connect()
    {
        conv::ConversationConfig cfg{};
        // User-set system prompt (over Wi-Fi) wins; empty falls back to the
        // firmware's built-in persona.
        cfg.instructions = system_prompt_.empty() ? kInstructions : system_prompt_;
        // Extra HTTP headers (e.g. Cloudflare Access token) for the WS upgrade.
        cfg.extra_headers = extra_headers_;
        if (provider_ == config::Provider::Gemini) {
            // Gemini Live model + voice. The model is namespaced as
            // "models/..."; the client prepends that for us when missing.
            // gemini-2.0-flash-live-001 was deprecated; the native-audio
            // preview is what's actually live on the Developer API right now.
            cfg.model = "gemini-2.5-flash-native-audio-preview-12-2025";
            cfg.voice = "Aoede"; // pre-built voice name; OK to leave empty
        } else {
            cfg.model = CONFIG_STACKCHAN_OPENAI_REALTIME_MODEL;
            cfg.voice = CONFIG_STACKCHAN_OPENAI_VOICE;
        }
        cfg.input_sample_rate_hz = mic_sample_rate_;
        cfg.output_sample_rate_hz = speaker_sample_rate_;
        // XiaoZhi v1 has no client-side tools — the server owns the persona and
        // drives the avatar's expression via its `llm` emotion messages, which
        // the client maps to AssistantEmotion events.
        if (provider_ != config::Provider::XiaoZhi) {
            cfg.tools.push_back(make_set_expression_tool());
            cfg.tools.push_back(make_set_head_pose_tool());
            cfg.tools.push_back(make_speak_katakoto_tool());
        }
        config_ = cfg;

        set_local(Local::Init);
        auto r = client_->start(config_);
        if (!r) {
            ESP_LOGE(kTag, "client start failed: %d", static_cast<int>(r.error()));
            return false;
        }
        return true;
    }

    bool wait_for_wifi_for_reconnect(bool allow_audio_abort)
    {
        if (wifi_is_connected()) {
            return true;
        }
        ESP_LOGW(kTag, "Wi-Fi disconnected; waiting before reconnecting the session");
        state_.conv.status.store(ConvStatus::WaitingWifi, std::memory_order_relaxed);
        while (!wifi_is_connected()) {
            if (allow_audio_abort &&
                state_.audio_stream_active.load(std::memory_order_acquire)) {
                ESP_LOGI(kTag, "session reconnect paused: BLE audio in use");
                return false;
            }
            vTaskDelay(pdMS_TO_TICKS(500));
        }
        ESP_LOGI(kTag, "Wi-Fi back; continuing to reconnect the session");
        return true;
    }

    bool delay_reconnect(std::uint32_t delay_ms, bool allow_audio_abort)
    {
        std::uint32_t waited_ms = 0;
        while (waited_ms < delay_ms) {
            if (allow_audio_abort &&
                state_.audio_stream_active.load(std::memory_order_acquire)) {
                ESP_LOGI(kTag, "session reconnect backoff paused: BLE audio in use");
                return false;
            }
            const std::uint32_t step_ms = std::min<std::uint32_t>(500, delay_ms - waited_ms);
            vTaskDelay(pdMS_TO_TICKS(step_ms));
            waited_ms += step_ms;
        }
        return true;
    }

    bool connect_with_retries(const char* context, bool allow_audio_abort)
    {
        unsigned failed_starts = 0;
        for (;;) {
            if (allow_audio_abort &&
                state_.audio_stream_active.load(std::memory_order_acquire)) {
                ESP_LOGI(kTag, "%s paused: BLE audio in use", context);
                return false;
            }
            if (!wait_for_wifi_for_reconnect(allow_audio_abort)) {
                return false;
            }

            ESP_LOGI(kTag, "%s: transport start attempt %u",
                     context, failed_starts + 1);
            if (connect()) {
                ESP_LOGI(kTag, "%s: transport start accepted", context);
                return true;
            }

            ++failed_starts;
            state_.conv.status.store(ConvStatus::Error, std::memory_order_relaxed);
            flush_events();
            const std::uint32_t backoff_ms = reconnect_backoff_ms(failed_starts);
            ESP_LOGE(kTag, "%s: transport start attempt %u failed; retrying in %u ms",
                     context, failed_starts, static_cast<unsigned>(backoff_ms));
            state_.conv.status.store(ConvStatus::Reconnecting, std::memory_order_relaxed);
            if (!delay_reconnect(backoff_ms, allow_audio_abort)) {
                return false;
            }
        }
    }

    // from_failure = false: clean handoff (e.g. Gemini goAway). Skip the
    // exponential backoff counter bump, but still go through the standard
    // reconnect flow (teardown old client + dma settle + connect).
    void recover_after_error(bool from_failure = true)
    {
        if (from_failure) {
            ++consecutive_recover_failures_;
            // capped at 30 s. After the n-th failure the backoff is 500 ms × 2^(n-1), i.e.
            // 500 / 1k / 2k / 4k / 8k / 16k / 30k. The first failure adds no wait here,
            // so a brief glitch does not cause a noticeable delay.
            const std::uint32_t backoff_ms =
                reconnect_backoff_ms(consecutive_recover_failures_);
            ESP_LOGW(kTag, "session recovery: attempt %u, backoff %u ms",
                     consecutive_recover_failures_, static_cast<unsigned>(backoff_ms));
            if (consecutive_recover_failures_ > 1) {
                // Skip the wait on the first try so transient blips don't
                // visibly delay reconnect; back off only when failures
                // pile up.
                if (!delay_reconnect(backoff_ms, /*allow_audio_abort=*/true)) {
                    return;
                }
            }
        } else {
            ESP_LOGI(kTag, "session handover (goAway)");
        }
        state_.conv.status.store(ConvStatus::Reconnecting, std::memory_order_relaxed);
        state_.conv.reconnects.fetch_add(1, std::memory_order_relaxed);
        if (local_ == Local::Speaking) {
            M5.Speaker.end();
            vTaskDelay(kI2sSettle);
        }
        state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
        client_->stop();
        // The shutdown fires WEBSOCKET_EVENT_DISCONNECTED (and sometimes a
        // trailing ERROR) which the trampoline turns into transport-error
        // events. We sit out the WS task's teardown, then flush the queue —
        // otherwise the next drain_events() picks up that disconnect and
        // treats it as a fresh failure of the new client, looping us back
        // into recovery.
        //
        // Wait until internal RAM has recovered before reconnecting. The
        // failure mode this catches: TLS handshake on a fresh WS allocates
        // a DMA descriptor for esp-aes; if we restart too fast after a
        // failed session the previous session's TLS state is still being
        // freed in the background, internal heap is fragmented, and the
        // alloc fails ("esp-aes: Failed to allocate memory" / SSL write
        // error).
        //
        // The metric that matters is the largest free *DMA-capable* block,
        // NOT MALLOC_CAP_INTERNAL: esp-aes allocates its DMA descriptor /
        // bounce buffer from MALLOC_CAP_DMA, a subset of internal SRAM.
        // We were gating on INTERNAL (which showed 26 KiB) while the DMA
        // pool was actually starved — so the gate "passed" and the AES
        // alloc still failed, looping forever.
        constexpr std::size_t kMinDmaLargestB = 16 * 1024;
        constexpr std::uint32_t kBackoffStepMs = 500;
        constexpr int kMaxBackoffSteps = 20; // 10 s cap
        for (int i = 0; i < kMaxBackoffSteps; ++i) {
            // Bail immediately if a BLE audio stream wants the stage —
            // the main loop's yield path will stop us cleanly and there's
            // no point reconnecting just to be torn down. This also keeps
            // the reconnect storm from stealing CPU / radio from playback.
            if (state_.audio_stream_active.load(std::memory_order_acquire)) {
                ESP_LOGI(kTag, "recover aborted — BLE audio active");
                return;
            }
            vTaskDelay(pdMS_TO_TICKS(kBackoffStepMs));
            const std::size_t dma_largest = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
            if (i == 0 || (i % 4) == 3) {
                ESP_LOGI(kTag,
                         "recover wait: INT free=%u largest=%u DMA-largest=%u  PSRAM free=%u",
                         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
                         static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
                         static_cast<unsigned>(dma_largest),
                         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
            }
            if (dma_largest >= kMinDmaLargestB && i >= 3) break; // min 2 s settle
        }
        flush_events();
        assistant_pcm_.clear();
        assistant_text_.clear();
        if (state_.audio_stream_active.load(std::memory_order_acquire)) return;
        (void)connect_with_retries("reconnect after an error", /*allow_audio_abort=*/true);
    }

    // ---- per-state servicing ----------------------------------------------

    // Single point of truth for the local state — also publishes "idle" so
    // demo_loop knows when it may run its idle behaviours (head poses,
    // nadenade) without fighting an in-progress reply.
    void set_local(Local s)
    {
        local_ = s;
        state_.conv.idle.store(s == Local::Listening, std::memory_order_relaxed);
        using avatar::VoiceState;
        VoiceState vs = VoiceState::Idle;
        switch (s) {
        case Local::Listening:
            // Session-ready standby is NOT the listening face: mapping it to
            // VoiceState::Listening would pin the face forever and make idle
            // decay unreachable. The listening face comes from real listening
            // moments instead — the wake-word overlay (asr_probe) and the
            // SpeechStarted event below; standby stays Idle so decay can run.
            vs = VoiceState::Idle;
            break;
        case Local::Thinking:
            vs = VoiceState::Thinking;
            break;
        case Local::Speaking:
            vs = VoiceState::Speaking;
            break;
        case Local::Init:
        case Local::Yielded:
            vs = VoiceState::Idle;
            break;
        }
        state_.set_voice_state(vs);
        // Mask servo motion for the entire reply playback (Speaking). Deriving
        // it from the state — rather than the speaker's isPlaying() — keeps the
        // head still across the brief silences between streamed reply segments,
        // which is exactly where the servo was twitching and cutting the audio.
        state_.servo.masked.store(s == Local::Speaking, std::memory_order_relaxed);
        if (s != Local::Speaking) {
            // A head command that arrived during the reply is applied now.
            if (auto head = head_commands_.finish()) apply_head(*head);
        }
        ConvStatus cs = ConvStatus::Connecting;
        switch (s) {
        case Local::Init: cs = ConvStatus::Connecting; break;
        case Local::Listening: cs = ConvStatus::Listening; break;
        case Local::Thinking:
        case Local::Speaking: cs = ConvStatus::Talking; break;
        case Local::Yielded: cs = ConvStatus::Yielded; break;
        }
        state_.conv.status.store(cs, std::memory_order_relaxed);
    }

    void service_state()
    {
        switch (local_) {
        case Local::Init:
            vTaskDelay(pdMS_TO_TICKS(20));
            break;
        case Local::Listening:
            service_mic();
            break;
        case Local::Thinking:
            if (!thinking_face_released_ && now_ms() - thinking_since_ms_ > kThinkingFaceMs) {
                thinking_face_released_ = true;
                state_.set_voice_state(avatar::VoiceState::Idle);
                ESP_LOGI(kTag, "thinking > %u ms with no reply — face back to idle, still waiting",
                         static_cast<unsigned>(kThinkingFaceMs));
            }
            if (now_ms() - thinking_since_ms_ > kThinkingTimeoutMs) {
                // The spec keeps the thinking face for the whole wait. A slow
                // but alive request keeps waiting (with a periodic warning);
                // only a dead session recovers here — transport errors also
                // recover through their own event paths.
                if (!state_.conv.active.load(std::memory_order_relaxed)) {
                    ESP_LOGW(kTag, "thinking with dead session → recover");
                    recover_after_error(/*from_failure=*/true);
                } else {
                    ESP_LOGW(kTag, "thinking > %u ms, request still in flight — keep waiting",
                             static_cast<unsigned>(kThinkingTimeoutMs));
                    thinking_since_ms_ = now_ms();
                }
            } else {
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            break;
        case Local::Speaking:
            // Belt-and-braces timeout: if playback has drained and nothing
            // has moved for kSpeakingStallTimeoutMs, force the same path
            // finish_speaking() would have taken once audio_complete_
            // flipped. Otherwise a missed generationComplete (e.g. Gemini
            // Live `interrupted` without the fall-through) leaves us stuck
            // here forever with mouth_open frozen at 0. See issue #2.
            if (!audio_complete_ && seg_pos_ >= assistant_pcm_.size() &&
                M5.Speaker.isPlaying(kSpeakerChannel) == 0 &&
                now_ms() - last_audio_progress_ms_ > kSpeakingStallTimeoutMs) {
                ESP_LOGW(kTag, "speaking timed out (stalled %u ms, %u ms total), forcing finish",
                         static_cast<unsigned>(now_ms() - last_audio_progress_ms_),
                         static_cast<unsigned>(now_ms() - playback_start_ms_));
                finish_speaking();
            } else {
                service_playback();
            }
            break;
        case Local::Yielded:
            // Handled at the top of run() — we should never actually
            // dispatch on this state, but the compiler insists.
            vTaskDelay(pdMS_TO_TICKS(100));
            break;
        }
    }

    // Stream the reply out through the internal-RAM segment ring. assistant_pcm_
    // may still be growing (streaming playback) — feed M5.Speaker while it has
    // a free slot (2-deep queue) and unplayed samples exist. Copying a segment
    // is a fast bulk memcpy, unlike the speaker's sample-by-sample resampler
    // read. Non-blocking so mouth-sync and barge-in polling keep ticking.
    void service_playback()
    {
        // Barge-in. Two triggers, both stop the reply and return to listening
        // (the mic is physically off while speaking, so these are the only ways
        // to interrupt on the half-duplex CoreS3 hardware):
        //  - an LCD screen tap (set by demo_loop) — the intended interrupt now
        //    that voice input is paused for the whole assistant turn;
        //  - a firm touch on the head sensor (legacy). firmly_touched() guards
        //    against stray Level-1 RFI blips that would clobber every reply.
        if (state_.barge_in_request.exchange(false, std::memory_order_relaxed) ||
            (touch_ != nullptr && touch_->read().firmly_touched())) {
            barge_in();
            return;
        }

        // Sample buffer-occupancy + speaker queue depth once per service tick,
        // before any topup decisions. spk_queue=2 = healthy steady, 0 = under-
        // run risk; pcm_lag = unplayed samples already in assistant_pcm_.
        spk_queue_.record(static_cast<float>(M5.Speaker.isPlaying(kSpeakerChannel)));
        pcm_lag_samples_.record(static_cast<float>(assistant_pcm_.size() - seg_pos_));

        while (seg_pos_ < assistant_pcm_.size() &&
               M5.Speaker.isPlaying(kSpeakerChannel) < kSegmentBuffers - 1) {
            const std::size_t n = std::min(kSegmentSamples, assistant_pcm_.size() - seg_pos_);
            std::memcpy(seg_buf_[seg_next_], assistant_pcm_.data() + seg_pos_, n * sizeof(std::int16_t));
            M5.Speaker.playRaw(seg_buf_[seg_next_], n, speaker_sample_rate_, /*stereo=*/false,
                               /*repeat=*/1, kSpeakerChannel, /*stop_current_sound=*/false);
            seg_pos_ += n;
            seg_next_ = (seg_next_ + 1) % kSegmentBuffers;
            last_audio_progress_ms_ = now_ms();
            // Account for every chunk whose tail is now queued. A single
            // playRaw can flush multiple chunks if they arrived faster than
            // service_playback could drain them, hence the while-loop here.
            while (!pending_chunks_.empty() &&
                   pending_chunks_.front().end_sample_offset <= seg_pos_) {
                recv_to_queued_ms_.record(
                    static_cast<float>(now_us() - pending_chunks_.front().recv_us) / 1000.0f);
                pending_chunks_.pop_front();
            }
        }

        trim_played_pcm();
        update_mouth();

        // Done only once every chunk has arrived, every sample has been queued,
        // and the speaker has physically drained.
        if (audio_complete_ && seg_pos_ >= assistant_pcm_.size() &&
            M5.Speaker.isPlaying(kSpeakerChannel) == 0) {
            finish_speaking();
        } else {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }

    // Drop already-queued samples from the front of assistant_pcm_, keeping
    // the last mouth-envelope window so update_mouth() still has data. Every
    // pending chunk ends past seg_pos_ (the rest were popped above), so its
    // offset stays positive after the shift.
    void trim_played_pcm()
    {
        const std::size_t window = speaker_sample_rate_ * kEnvelopeStepMs / 1000u;
        if (seg_pos_ < speaker_sample_rate_ * kPcmTrimAfterMs / 1000u) return;
        const std::size_t drop = seg_pos_ - window;
        assistant_pcm_.erase(assistant_pcm_.begin(), assistant_pcm_.begin() + static_cast<std::ptrdiff_t>(drop));
        seg_pos_ -= drop;
        pcm_trimmed_samples_ += drop;
        for (auto& chunk : pending_chunks_) {
            chunk.end_sample_offset -= drop;
        }
    }

    void barge_in()
    {
        ESP_LOGI(kTag, "barge-in: user touched the head, interrupting reply");
        M5.Speaker.stop();
        (void)client_->cancel_response();
        state_.set_balloon_text("我在呢？", /*hold_ms=*/1500);
        log_playback_metrics();
        enter_listening();
    }

    // Double-buffered mic streaming. M5.Mic keeps a 2-deep queue; when one
    // slot frees, push that chunk upstream and re-queue it.
    // Listening ripple: smoothed loudness of the mic chunk just captured, for
    // the face DSL's voice_level. Fast attack so the halo jumps when you speak,
    // slow release so it does not flicker between syllables.
    void update_voice_level(std::span<const std::int16_t> pcm)
    {
        if (pcm.empty()) return;
        std::int64_t sum = 0;
        for (const std::int16_t sample : pcm) {
            sum += static_cast<std::int32_t>(sample) * sample;
        }
        const float rms = std::sqrt(static_cast<float>(sum) / static_cast<float>(pcm.size()));
        // CoreS3 mic: room noise sits around 100-200 RMS, normal speech 800-4000.
        constexpr float kNoiseFloor = 150.0f;
        constexpr float kFullScale = 3500.0f;
        const float level = std::sqrt(std::clamp((rms - kNoiseFloor) / kFullScale, 0.0f, 1.0f));
        const float prev = state_.face.voice_level.load(std::memory_order_relaxed);
        const float next = level > prev ? level : prev * 0.85f + level * 0.15f;
        state_.face.voice_level.store(next, std::memory_order_relaxed);
    }

    void service_mic()
    {
        if (M5.Mic.isRecording() < 2) {
            const auto& buf = mic_buf_[mic_read_];
            update_voice_level(std::span<const std::int16_t>{buf.data(), buf.size()});
            (void)client_->push_audio(std::span<const std::int16_t>{buf.data(), buf.size()});
            M5.Mic.record(mic_buf_[mic_read_].data(), mic_buf_[mic_read_].size(), mic_sample_rate_, /*stereo=*/false);
            mic_read_ ^= 1;
        } else {
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }

    void enter_listening()
    {
        M5.Speaker.end();
        vTaskDelay(kI2sSettle);
        state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
        // Safety net: log_playback_metrics already cleared these on
        // finish_speaking / barge_in, but recover_after_error skips that
        // path. Clear here so a stale pending entry from a half-finished
        // turn doesn't corrupt the next turn's recv_to_queued_ms.
        pending_chunks_.clear();
        assistant_pcm_.clear();
        // Reserve ahead so the streaming-playback inserts don't keep
        // reallocating the PSRAM buffer as the reply grows.
        assistant_pcm_.reserve(speaker_sample_rate_ * 20); // ~20 s headroom
        assistant_text_.clear();
        audio_complete_ = false;
        // Drop any barge-in tap that arrived during the just-finished turn so it
        // can't immediately re-trigger now that we're listening again.
        state_.barge_in_request.store(false, std::memory_order_relaxed);
        // Prime the 2-deep mic queue.
        M5.Mic.record(mic_buf_[0].data(), mic_buf_[0].size(), mic_sample_rate_, /*stereo=*/false);
        M5.Mic.record(mic_buf_[1].data(), mic_buf_[1].size(), mic_sample_rate_, /*stereo=*/false);
        mic_read_ = 0;
        set_local(Local::Listening);
    }

    // Switch the I2S bus to the speaker and begin streaming playback. Called
    // as soon as the jitter buffer fills (or on AssistantAudioDone for replies
    // shorter than the jitter buffer) — assistant_pcm_ may still be growing.
    void start_speaking()
    {
        state_.face.voice_level.store(0.0f, std::memory_order_relaxed);
        M5.Mic.end();
        vTaskDelay(kI2sSettle);
        seg_pos_ = 0;
        seg_next_ = 0;
        pcm_trimmed_samples_ = 0;
        playback_start_ms_ = now_ms();
        last_audio_progress_ms_ = playback_start_ms_;
        playback_start_us_ = now_us();
        set_local(Local::Speaking);
        ESP_LOGI(kTag, "speaking (streaming)");
    }

    // Mouth-open is derived on the fly from the most-recently-queued 16 ms
    // window of reply audio — no precomputed envelope, so it works while
    // assistant_pcm_ is still being streamed in. We use seg_pos_ (the
    // sample-accurate queue cursor) rather than wall-clock elapsed so the
    // envelope tracks actual playback even when the stream falls behind
    // real-time. The old wall-clock derivation silently stuck the mouth at
    // 0 whenever the network/decoder couldn't keep up with realtime: once
    // (elapsed/step)*window overtook assistant_pcm_.size(), the
    // out-of-bounds guard short-circuited and the mouth never moved again
    // for the rest of the reply, even though the speaker kept playing.
    // seg_pos_ is bounded by assistant_pcm_.size() in service_playback,
    // so the window we read is always populated.
    //
    // The window we sample ends at seg_pos_ (newest queued samples) rather
    // than where DMA is physically playing — that lags by up to
    // 2 × kSegmentSamples (≈30–50 ms), which is below the perceptual
    // threshold and avoids needing a DMA position read out of M5.Speaker.
    void update_mouth()
    {
        const std::size_t window = speaker_sample_rate_ * kEnvelopeStepMs / 1000u;
        if (seg_pos_ < window) {
            state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
            return;
        }
        const std::size_t begin = seg_pos_ - window;
        std::int32_t peak = 0;
        for (std::size_t i = begin; i < begin + window; ++i) {
            peak = std::max(peak, std::abs(static_cast<std::int32_t>(assistant_pcm_[i])));
        }
        state_.face.mouth_open.store(static_cast<float>(peak) / 32767.0f, std::memory_order_relaxed);
    }

    void finish_speaking()
    {
        M5.Speaker.end();
        vTaskDelay(kI2sSettle);
        state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
        log_playback_metrics();
        enter_listening();
    }

    // Single-line summary of the just-finished turn: queue latency, end-to-end
    // delay, buffer health. Logged right before we drop back to Listening so
    // each turn produces exactly one metric line, easy to grep / diff against
    // earlier turns when chasing cross-turn degradation.
    void log_playback_metrics()
    {
        const auto count = recv_to_queued_ms_.count;
        const auto pcm_lag_max = pcm_lag_samples_.max();
        // Effective dispatch rate: pcm_trimmed_samples_ + seg_pos_ counts
        // samples handed to playRaw (seg_pos_ alone restarts after a trim),
        // which is queue-bound to <= 2 segments behind actual DMA output.
        // Over a turn >>341 ms (typical), the playRaw rate ≈ the physical
        // playback rate. If this comes out materially below speaker_sample_rate_
        // (24000 for Gemini), the I2S clock divisor isn't producing the rate
        // we asked for — which would explain monotonic buffer growth.
        const std::int64_t elapsed_us = playback_start_us_ > 0
                                            ? (now_us() - playback_start_us_)
                                            : 0;
        const double played_sps = elapsed_us > 0
                                      ? static_cast<double>(pcm_trimmed_samples_ + seg_pos_) * 1.0e6 / static_cast<double>(elapsed_us)
                                      : 0.0;
        ESP_LOGI(kTag,
                 "metrics(play): chunks=%u  recv_lag_us avg=%.0f min=%.0f max=%.0f  "
                 "recv_to_queued_ms avg=%.1f min=%.1f max=%.1f  "
                 "spk_q avg=%.2f min=%.0f max=%.0f  pcm_lag_samples avg=%.0f max=%.0f  "
                 "played_sps=%.1f (nominal=%u)",
                 static_cast<unsigned>(count),
                 recv_lag_us_.mean(), static_cast<double>(recv_lag_us_.min()),
                 static_cast<double>(recv_lag_us_.max()),
                 recv_to_queued_ms_.mean(), static_cast<double>(recv_to_queued_ms_.min()),
                 static_cast<double>(recv_to_queued_ms_.max()),
                 spk_queue_.mean(), static_cast<double>(spk_queue_.min()),
                 static_cast<double>(spk_queue_.max()),
                 pcm_lag_samples_.mean(), static_cast<double>(pcm_lag_max),
                 played_sps, static_cast<unsigned>(speaker_sample_rate_));
        // Publish the snapshot so the HTTP / BLE settings endpoints can serve
        // it (e.g. /api/metrics/audio polled from a phone). Writes go via
        // SharedState's mutex so a poll mid-write sees a coherent struct.
        SharedState::AudioMetrics snap{};
        snap.turn_at_ms = now_ms();
        snap.chunk_count = static_cast<std::uint32_t>(count);
        snap.speaker_sample_rate = speaker_sample_rate_;
        snap.recv_lag_us_avg = static_cast<float>(recv_lag_us_.mean());
        snap.recv_lag_us_min = recv_lag_us_.min();
        snap.recv_lag_us_max = recv_lag_us_.max();
        snap.recv_to_queued_ms_avg = static_cast<float>(recv_to_queued_ms_.mean());
        snap.recv_to_queued_ms_min = recv_to_queued_ms_.min();
        snap.recv_to_queued_ms_max = recv_to_queued_ms_.max();
        snap.spk_queue_avg = static_cast<float>(spk_queue_.mean());
        snap.spk_queue_min = spk_queue_.min();
        snap.spk_queue_max = spk_queue_.max();
        snap.pcm_lag_samples_avg = static_cast<float>(pcm_lag_samples_.mean());
        snap.pcm_lag_samples_max = pcm_lag_samples_.max();
        snap.played_sps = static_cast<float>(played_sps);
        // Session-cumulative uplink congestion counter, maintained by the
        // backend client (per-chunk eviction warnings are rate-limited there).
        snap.tx_evicted_chunks = client_ != nullptr ? client_->tx_evicted_chunks() : 0;
        state_.update_audio_metrics(snap);
        recv_lag_us_.reset();
        recv_to_queued_ms_.reset();
        spk_queue_.reset();
        pcm_lag_samples_.reset();
        pending_chunks_.clear();
    }

    // ---- event handling ----------------------------------------------------

    void handle_event(const conv::ConversationEvent& ev)
    {
        switch (ev.type) {
        case conv::ConversationEventType::StateChanged:
            // The protocol reaching Listening for the first time means the
            // session is live. Later protocol transitions are ignored — the
            // local state machine owns mic/speaker timing.
            if (local_ == Local::Init && ev.state == conv::ConversationState::Listening) {
                ESP_LOGI(kTag, "session ready");
                state_.conv.active.store(true, std::memory_order_relaxed);
                enter_listening();
            }
            // Any time the backend reaches Listening (= setup accepted, session
            // alive) reset the consecutive-failure counter so a future
            // transient outage starts the backoff fresh instead of inheriting
            // the prior storm's tail.
            if (ev.state == conv::ConversationState::Listening &&
                consecutive_recover_failures_ > 0) {
                ESP_LOGI(kTag, "session healthy → reset recover counter (was %u)",
                         consecutive_recover_failures_);
                consecutive_recover_failures_ = 0;
            }
            break;

        case conv::ConversationEventType::GoingAway:
            // Server is warning us it will close soon (typically the ~15 min
            // audio-session cap; Gemini docs § "Session resumption"). Tear
            // down + reconnect proactively while we still have the
            // resumption handle cached — this rotates the underlying TCP/TLS
            // session without losing the conversation context. Not a
            // failure, so don't bump the backoff counter.
            ESP_LOGI(kTag, "GoingAway: timeLeft=%s → proactive reconnect", ev.text.c_str());
            recover_after_error(/*from_failure=*/false);
            break;

        case conv::ConversationEventType::SpeechStarted:
            ESP_LOGI(kTag, "user speech started");
            state_.note_face_activity();
            // The user is actually talking to us — this is the listening
            // window. SpeechStopped flips Local to Thinking which overwrites
            // it, and any later set_local returns the state to Idle.
            state_.set_voice_state(avatar::VoiceState::Listening);
            break;

        case conv::ConversationEventType::SpeechStopped:
            ESP_LOGI(kTag, "user speech stopped");
            if (local_ == Local::Listening) {
                set_local(Local::Thinking);
                thinking_since_ms_ = now_ms();
                thinking_face_released_ = false;
            }
            break;

        case conv::ConversationEventType::UserTranscript:
            ESP_LOGI(kTag, "user: %s", ev.text.c_str());
            state_.note_face_activity();
            if (!ev.text.empty()) {
                state_.set_balloon_text(ev.text, /*hold_ms=*/2500);
            }
            break;

        case conv::ConversationEventType::AssistantTextDelta:
            assistant_text_ += ev.text;
            break;

        case conv::ConversationEventType::AssistantTextDone:
            if (!ev.text.empty()) {
                assistant_text_ = ev.text;
            }
            ESP_LOGI(kTag, "assistant: %s", assistant_text_.c_str());
            if (!assistant_text_.empty()) {
                state_.set_balloon_text(assistant_text_, /*hold_ms=*/0);
            }
            break;

        case conv::ConversationEventType::AssistantAudioChunk:
            // Ignore late chunks for a turn we already abandoned (barge-in).
            if (ev.audio && (local_ == Local::Thinking || local_ == Local::Speaking)) {
                const std::int64_t recv = now_us();
                if (ev.emit_us > 0) {
                    recv_lag_us_.record(static_cast<float>(recv - ev.emit_us));
                }
                assistant_pcm_.insert(assistant_pcm_.end(), ev.audio->begin(), ev.audio->end());
                pending_chunks_.push_back({assistant_pcm_.size(), recv});
                last_audio_progress_ms_ = now_ms();
                // Streaming: start playback as soon as the jitter buffer fills,
                // rather than waiting for the whole reply.
                if (local_ == Local::Thinking && assistant_pcm_.size() >= jitter_buffer_samples_) {
                    tool_pending_ = false;
                    start_speaking();
                }
            }
            break;

        case conv::ConversationEventType::AssistantAudioDone:
            audio_complete_ = true;
            // Reply shorter than the jitter buffer — it never tripped the
            // streaming start, so begin playback now.
            if (local_ == Local::Thinking && !assistant_pcm_.empty()) {
                tool_pending_ = false;
                start_speaking();
            }
            break;

        case conv::ConversationEventType::AssistantEmotion:
            // Backend-reported affect (XiaoZhi `llm.emotion`). Maps straight to
            // an avatar expression — no tool round-trip, so it never touches
            // tool_pending_ / the turn state machine.
            if (const auto expr = parse_emotion(ev.text.c_str())) {
                ESP_LOGI(kTag, "emotion: %s", ev.text.c_str());
                state_.face.expression.store(static_cast<int>(*expr), std::memory_order_relaxed);
                state_.note_face_activity();
            }
            break;

        case conv::ConversationEventType::ToolCallRequested:
            if (ev.tool_call) {
                dispatch_tool(*ev.tool_call);
            }
            break;

        case conv::ConversationEventType::ResponseDone:
            // The response that carried a tool call ends here, but a follow-up
            // response (the model's actual reply) is still coming after we
            // submit the tool result — stay in Thinking and wait for it.
            if (tool_pending_) {
                tool_pending_ = false;
                thinking_since_ms_ = now_ms();
                thinking_face_released_ = false;
            } else if (local_ == Local::Thinking && assistant_pcm_.empty()) {
                // Text-only or empty turn: nothing will switch us to Speaking.
                enter_listening();
            }
            break;

        case conv::ConversationEventType::Error: {
            // Transport errors (disconnect / handshake) are fatal — the
            // session is gone and must be rebuilt. Server-side errors (bad
            // request, a no-op cancellation, rate-limit notices, …) leave the
            // session alive, so just log them and carry on.
            const bool transport_error = ev.error == conv::ConversationError::NotConnected ||
                                         ev.error == conv::ConversationError::TransportInit;
            if (transport_error) {
                ESP_LOGE(kTag, "transport error: %s — reconnecting", ev.text.c_str());
                state_.set_balloon_text("连接错误", /*hold_ms=*/3000);
                recover_after_error();
            } else {
                ESP_LOGW(kTag, "server error (non-fatal): %s", ev.text.c_str());
            }
            break;
        }

        case conv::ConversationEventType::HeadPose: {
            const conv::HeadCommand head{clamp_deg(ev.head_yaw, limits_.yaw_min_deg, limits_.yaw_max_deg),
                                         clamp_deg(ev.head_pitch, limits_.pitch_min_deg, limits_.pitch_max_deg),
                                         ev.head_speed};
            if (auto ready = head_commands_.receive(head, local_ == Local::Speaking)) apply_head(*ready);
            break;
        }

        case conv::ConversationEventType::LedColor: {
            const auto solid = led_runtime::from_rgb(ev.led_r, ev.led_g, ev.led_b);
            state_.led.color.store(solid.color, std::memory_order_relaxed);
            // A XiaoZhi LED message is a temporary status light, not a user setting: only runtime state is
            // written here; the next led_task tick drives the real Board::LedStrip. Nothing goes to NVS.
            state_.led.mode.store(solid.mode, std::memory_order_relaxed);
            state_.led.brightness.store(solid.brightness, std::memory_order_relaxed);
            ESP_LOGI(kTag, "LED status light: r=%u g=%u b=%u",
                     static_cast<unsigned>(ev.led_r),
                     static_cast<unsigned>(ev.led_g),
                     static_cast<unsigned>(ev.led_b));
            break;
        }
        }
    }

    // Gateway head command (face tracking, move_head on voice-only firmware).
    // Holding off the idle poses for 2 s keeps demo_loop from replacing the
    // pose before the next command arrives.
    void apply_head(const conv::HeadCommand& head)
    {
        state_.servo.speed_override.store(head.speed, std::memory_order_relaxed);
        state_.servo.target_yaw_deg.store(head.yaw, std::memory_order_relaxed);
        state_.servo.target_pitch_deg.store(head.pitch, std::memory_order_relaxed);
        state_.servo.head_hold_until_ms.store(now_ms() + 2000, std::memory_order_relaxed);
    }

    // ---- tool dispatch -----------------------------------------------------

    void dispatch_tool(const conv::ToolCall& call)
    {
        ESP_LOGI(kTag, "tool call: %s args=%s", call.name.c_str(), call.arguments_json.c_str());
        std::string result = R"({"ok":false,"error":"unknown tool"})";

        if (call.name == "set_expression") {
            result = handle_set_expression(call.arguments_json);
        } else if (call.name == "set_head_pose") {
            result = handle_set_head_pose(call.arguments_json);
        } else if (call.name == "speak_katakoto") {
            result = handle_speak_katakoto(call.arguments_json);
        }

        // The model will issue a follow-up response after we return the tool
        // result; mark it pending so ResponseDone for the tool-call response
        // doesn't drop us back to Listening prematurely.
        tool_pending_ = true;
        auto r = client_->submit_tool_result(call.call_id, result);
        if (!r) {
            ESP_LOGE(kTag, "submit_tool_result failed: %d", static_cast<int>(r.error()));
        }
    }

    std::string handle_set_expression(const std::string& args)
    {
        cJSON* root = cJSON_Parse(args.c_str());
        if (root == nullptr) {
            return R"({"ok":false,"error":"bad arguments"})";
        }
        const cJSON* item = cJSON_GetObjectItemCaseSensitive(root, "expression");
        const char* name = cJSON_IsString(item) ? item->valuestring : nullptr;
        const auto expr = parse_expression(name);
        std::string result;
        if (expr) {
            state_.face.expression.store(static_cast<int>(*expr), std::memory_order_relaxed);
            state_.note_face_activity();
            result = R"({"ok":true})";
        } else {
            result = R"({"ok":false,"error":"unknown expression"})";
        }
        cJSON_Delete(root);
        return result;
    }

    // Synthesise the kana phrase through jtts and play it through the speaker.
    // Blocks the conversation task until playback drains, so the tool result
    // is sent back only after the audible utterance finishes — this avoids
    // overlapping with any follow-up reply the model might produce. The PCM
    // is generated into a PSRAM vector then streamed through seg_buf_ (the
    // same internal-RAM ring used by reply playback) to dodge PSRAM contention.
    std::string handle_speak_katakoto(const std::string& args)
    {
        cJSON* root = cJSON_Parse(args.c_str());
        if (root == nullptr) {
            return R"({"ok":false,"error":"bad arguments"})";
        }
        const cJSON* item = cJSON_GetObjectItemCaseSensitive(root, "kana");
        std::string kana_utf8;
        if (cJSON_IsString(item) && item->valuestring != nullptr) {
            kana_utf8 = item->valuestring;
        }
        cJSON_Delete(root);
        if (kana_utf8.empty()) {
            return R"({"ok":false,"error":"kana required"})";
        }

        const std::u32string kana = decode_utf8(kana_utf8);
        if (kana.empty()) {
            return R"({"ok":false,"error":"invalid utf8"})";
        }

        // Robotic-katakoto preset: low monotone male voice, slightly halting
        // mora pace. Deliberately different from the assistant's normal voice
        // so the user clearly hears it as a separate "mode".
        constexpr std::uint32_t kKatakotoRate = 16000;
        stackchan::jtts::Options opt;
        opt.voice = stackchan::jtts::Voice::Male;
        opt.f0_hz = 140.0f;
        opt.mora_ms = 140.0f;
        opt.gain = 0.8f;
        opt.sample_rate_hz = kKatakotoRate;

        std::vector<std::int16_t> pcm;
        auto r = stackchan::jtts::synthesize(kana, pcm, opt);
        if (!r) {
            ESP_LOGW(kTag, "jtts synthesize failed: %s",
                     stackchan::jtts::to_string(r.error()));
            return R"({"ok":false,"error":"synthesize failed"})";
        }
        if (pcm.empty()) {
            return R"({"ok":true,"warning":"empty audio"})";
        }

        // I2S handoff: mic → speaker. We may be in Listening (mic primed) or
        // Thinking (mic primed but not being drained); either way the bus
        // needs to belong to the speaker before playRaw.
        M5.Mic.end();
        vTaskDelay(kI2sSettle);

        // Pre-compute peak envelope so the avatar mouth opens in sync with the
        // utterance even though we're not running the normal service_playback.
        const std::size_t env_window = kKatakotoRate * kEnvelopeStepMs / 1000u;
        std::vector<float> envelope;
        if (env_window > 0) {
            envelope.reserve((pcm.size() + env_window - 1) / env_window);
            for (std::size_t i = 0; i < pcm.size(); i += env_window) {
                std::int32_t peak = 0;
                const std::size_t end = std::min(i + env_window, pcm.size());
                for (std::size_t j = i; j < end; ++j) {
                    peak = std::max(peak, std::abs(static_cast<std::int32_t>(pcm[j])));
                }
                envelope.push_back(static_cast<float>(peak) / 32767.0f);
            }
        }

        auto update_mouth_from_envelope = [&](std::uint32_t elapsed_ms) {
            const std::size_t idx = elapsed_ms / kEnvelopeStepMs;
            const float open = idx < envelope.size() ? envelope[idx] : 0.0f;
            state_.face.mouth_open.store(open, std::memory_order_relaxed);
        };

        const std::uint32_t start_ms = now_ms();
        std::size_t pos = 0;
        std::size_t next = 0;
        while (pos < pcm.size()) {
            if (M5.Speaker.isPlaying(kSpeakerChannel) < kSegmentBuffers - 1) {
                const std::size_t n = std::min(kSegmentSamples, pcm.size() - pos);
                std::memcpy(seg_buf_[next], pcm.data() + pos, n * sizeof(std::int16_t));
                M5.Speaker.playRaw(seg_buf_[next], n, kKatakotoRate, /*stereo=*/false,
                                   /*repeat=*/1, kSpeakerChannel, /*stop_current_sound=*/false);
                pos += n;
                next = (next + 1) % kSegmentBuffers;
            }
            update_mouth_from_envelope(now_ms() - start_ms);
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        while (M5.Speaker.isPlaying(kSpeakerChannel) > 0) {
            update_mouth_from_envelope(now_ms() - start_ms);
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        M5.Speaker.end();
        vTaskDelay(kI2sSettle);
        state_.face.mouth_open.store(0.0f, std::memory_order_relaxed);
        return R"({"ok":true})";
    }

    std::string handle_set_head_pose(const std::string& args)
    {
        cJSON* root = cJSON_Parse(args.c_str());
        if (root == nullptr) {
            return R"({"ok":false,"error":"bad arguments"})";
        }
        const cJSON* yaw = cJSON_GetObjectItemCaseSensitive(root, "yaw_deg");
        const cJSON* pitch = cJSON_GetObjectItemCaseSensitive(root, "pitch_deg");
        std::string result;
        if (cJSON_IsNumber(yaw) && cJSON_IsNumber(pitch)) {
            const float yaw_deg = std::clamp(static_cast<float>(yaw->valuedouble), -40.0f, 40.0f);
            const float pitch_deg = std::clamp(static_cast<float>(pitch->valuedouble), -10.0f, 25.0f);
            state_.servo.target_yaw_deg.store(yaw_deg, std::memory_order_relaxed);
            state_.servo.target_pitch_deg.store(pitch_deg, std::memory_order_relaxed);
            result = R"({"ok":true})";
        } else {
            result = R"({"ok":false,"error":"yaw_deg and pitch_deg required"})";
        }
        cJSON_Delete(root);
        return result;
    }

    // ---- members -----------------------------------------------------------

    const char* provider_name() const
    {
        switch (provider_) {
        case config::Provider::Gemini: return "Gemini";
        case config::Provider::XiaoZhi: return "XiaoZhi";
        default: return "OpenAI";
        }
    }

    SharedState& state_;
    std::string api_key_;
    config::Provider provider_;
    board::Si12tTouch* touch_; // top touch sensor for barge-in (may be null)
    std::string xiaozhi_url_;
    std::string xiaozhi_token_;
    std::string system_prompt_;
    std::string extra_headers_;
    ServoLimits limits_;
    conv::HeadCommandQueue head_commands_;
    conv::ConversationConfig config_{};
    std::unique_ptr<conv::ConversationService> client_;
    QueueHandle_t event_queue_{nullptr};

    // Per-provider audio rates set in the constructor.
    std::uint32_t mic_sample_rate_{8000};
    std::uint32_t speaker_sample_rate_{8000};
    std::size_t mic_chunk_samples_{320};
    std::size_t jitter_buffer_samples_{2400};

    Local local_{Local::Init};
    std::uint32_t thinking_since_ms_{0};
    // The thinking face may already be back to idle (kThinkingFaceMs limit); reset on entering Thinking.
    bool thinking_face_released_{false};
    bool tool_pending_{false};

    std::array<std::vector<std::int16_t>, 2> mic_buf_{};
    int mic_read_{0};

    std::vector<std::int16_t> assistant_pcm_;          // unplayed reply audio, in PSRAM (played front is trimmed)
    bool audio_complete_{false};                       // every AssistantAudioChunk has arrived
    std::array<std::int16_t*, kSegmentBuffers> seg_buf_{}; // internal-RAM playback ring
    std::size_t seg_pos_{0};                           // next sample in assistant_pcm_ to play
    std::size_t seg_next_{0};                          // next ring slot to write
    std::size_t pcm_trimmed_samples_{0};               // samples already dropped from the front of assistant_pcm_
    std::uint32_t last_audio_progress_ms_{0};          // last chunk arrival or playRaw, for the stall timeout
    std::uint32_t playback_start_ms_{0};
    std::int64_t playback_start_us_{0};
    std::string assistant_text_;

    // Consecutive transport-failure count for exponential reconnect backoff.
    // Bumped on every recover_after_error(/*from_failure=*/true) and reset
    // when StateChanged → Listening fires for the new session (= the server
    // accepted setup). Capped backoff at kRecoverBackoffCapMs so prolonged
    // outages don't drag each retry past 30 s.
    unsigned consecutive_recover_failures_{0};

    // --- audio-pipeline diagnostics ----------------------------------------
    // A chunk's identity is "the last sample offset in assistant_pcm_ at the
    // time it was appended". When seg_pos_ advances past that offset, the
    // chunk has been handed to M5.Speaker.playRaw — we measure recv→queued
    // latency for each chunk this way and pop the entry off pending_.
    struct PendingChunk {
        std::size_t end_sample_offset; // assistant_pcm_.size() after append (shifted on trim)
        std::int64_t recv_us;          // now_us() when we got the AssistantAudioChunk event
    };
    std::deque<PendingChunk> pending_chunks_;
    // Recv lag (event-queue hop): receiver's now - producer's emit_us. Mostly
    // measures FreeRTOS-queue + WS-task scheduling latency.
    conv::Stats<float> recv_lag_us_;
    // End-to-end queueing: now - recv_us at the moment seg_pos_ advances past
    // this chunk's end_sample_offset. Captures how long a chunk sat in
    // assistant_pcm_ before the playback loop grabbed it.
    conv::Stats<float> recv_to_queued_ms_;
    // Speaker queue depth (0..2) sampled each service_playback iteration.
    // 0 = under-run risk, 2 = healthy steady state.
    conv::Stats<float> spk_queue_;
    // Samples already received but not yet handed to playRaw at each
    // service_playback iteration. A growing lag means playback can't keep up
    // with the arrival rate; a near-zero lag means the speaker is the
    // bottleneck (or we're caught up).
    conv::Stats<float> pcm_lag_samples_;
};

void conversation_task_entry(void* arg)
{
    auto& args = *static_cast<ConversationTaskArgs*>(arg);
    auto* coordinator = new Coordinator(*args.state, args.api_key, args.provider, args.touch,
                                        args.xiaozhi_url, args.xiaozhi_token, args.system_prompt,
                                        args.extra_headers, args.limits, args.seg_buf);
    coordinator->run();
    // run() only returns by deleting the task; keep the object alive regardless.
    // A task created WithCaps must be deleted WithCaps, or the PSRAM stack is not returned.
    vTaskDeleteWithCaps(nullptr);
}

} // namespace

void start_conversation_task(ConversationTaskArgs& args)
{
    // Stack in PSRAM to save internal RAM: this task is a TLS/WS read-write loop with no flash
    // writes (same as the say_worker / asr PSRAM stacks), so 8 KiB go back to internal RAM.
    constexpr UBaseType_t kCaps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    xTaskCreatePinnedToCoreWithCaps(conversation_task_entry, "conversation", 8192, &args, 5, nullptr,
                                    0, kCaps);
}

} // namespace stackchan::app
