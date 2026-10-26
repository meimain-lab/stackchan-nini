// SPDX-FileCopyrightText: 2026 Kenta IDA <fuga@fugafuga.org>
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include <M5GFX.h>

#include "avatar/canvas.hpp"
#include "avatar/draw_context.hpp"
#include "avatar/expression.hpp"
#include "avatar/face_tuning.hpp"
#include "avatar/palette.hpp"

namespace stackchan::avatar {

class Avatar {
public:
    Avatar();
    ~Avatar();

    Avatar(const Avatar&) = delete;
    Avatar& operator=(const Avatar&) = delete;
    Avatar(Avatar&&) noexcept;
    Avatar& operator=(Avatar&&) noexcept;

    void set_expression(Expression expression) noexcept;
    void set_voice_state(VoiceState voice) noexcept;
    // `hold_ms == 0` keeps the overlay until `clear_overlay`.
    void set_overlay(Expression overlay, std::uint32_t hold_ms) noexcept;
    void clear_overlay() noexcept;
    void note_activity() noexcept;
    Expression resolve_expression(std::uint32_t now_ms) noexcept;
    void set_mouth_open(float ratio) noexcept;
    // Live loudness of the user's voice while listening, 0..1 (DSL voice_level).
    void set_voice_level(float level) noexcept;
    void set_gaze(float horizontal, float vertical) noexcept;
    void set_palette(const Palette& palette) noexcept;
    // Rebuild the face layout from user tuning (eye/eyebrow/mouth geometry) and
    // apply its face/background colours. Takes effect on the next tick(); safe
    // to call live (e.g. from the render task on a config change).
    void set_face_tuning(const FaceTuning& tuning) noexcept;
    // Show `text` in the balloon. `hold_ms` overrides the default display
    // time (0 = use balloon defaults: short text holds for a few seconds,
    // long text plays one full marquee pass).
    void set_balloon_text(std::string_view text, std::uint32_t hold_ms = 0);
    void clear_balloon() noexcept;
    // True once the current balloon has been fully displayed (hold elapsed
    // or one marquee pass completed). Stays true until the next
    // set_balloon_text / clear_balloon.
    bool is_balloon_done() const noexcept;

    // Drives the animators with the current time (ms) and renders one frame
    // through `canvas` (begin_frame → face / effect / balloon). The canvas owns
    // the framebuffer/present strategy (buffered vs direct); Avatar never
    // touches the panel itself. Authored for a 320x240 surface.
    void tick(std::uint32_t now_ms, RichCanvas& canvas);

    // Force a full background repaint on the next tick(). Needed for the direct
    // (PSRAM-less) strategy after the panel was used by something else — e.g.
    // returning from the on-device UI. No-op cost on the buffered strategy.
    void request_full_repaint() noexcept;

    // aora eye-ring wiring: the caller (render task) owns a frame buffer of kRingVarCount floats
    // and fills it with the compositor (main/aora_face.hpp) before tick() every frame; this only
    // stores a non-owning pointer, which the VM reads through the Var::RingBase range. nullptr reads 0.
    void set_eye_ring_buffer(const float* buffer) noexcept;
    // The compositor reads the current expression weights / blink / gaze (values from the previous tick).
    const DrawContext& draw_context() const noexcept;

    // Hot-swap the face bytecode. `bytes` must hold an `AVDS` v1 file (see
    // assets/grok_face.avdsl / tools/avatar_dsl/). Returns true on success;
    // on failure the previous bytecode (or the firmware-embedded default) is
    // left in place and an ESP_LOG error describes the decode error. The buffer
    // is copied internally so the caller may free `bytes` immediately.
    bool load_face_bytecode(std::span<const std::uint8_t> bytes);
    // Revert to the firmware-embedded default face. Always succeeds (the
    // default bytecode is validated at link time).
    void reset_face_bytecode() noexcept;
    // True once the VM has decoded a face (embedded default or a hot-swap).
    bool has_face() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace stackchan::avatar
