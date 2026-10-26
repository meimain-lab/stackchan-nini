// SPDX-FileCopyrightText: 2026 Kenta IDA <fuga@fugafuga.org>
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "avatar/expression.hpp"
#include "avatar/palette.hpp"

namespace stackchan::avatar {

struct DrawContext {
    Expression expression{Expression::Neutral};
    // Source expression for a C++-owned ease. `expression_blend` is 0 at the
    // start of a transition (fully the from-pose) and 1 when idle / done
    // (fully `expression`). Two nested holds freeze interrupted mixes so a
    // retarget does not snap. The from-pose is
    // lerp(lerp(`expression_from`, `expression_hold_to`,
    //           `expression_hold_blend`),
    //      `expression_hold2_to`, `expression_hold2_blend`).
    // `expression_hold2_blend` is 0 when the second nest is idle.
    Expression expression_from{Expression::Neutral};
    float expression_blend{1.0f};
    Expression expression_hold_to{Expression::Neutral};
    float expression_hold_blend{1.0f};
    Expression expression_hold2_to{Expression::Neutral};
    float expression_hold2_blend{0.0f};
    float breath{0.0f};
    // External / commanded gaze target — written by Avatar::set_gaze (e.g.
    // touch-follow, future API gesture). Saccade is added on top so the
    // eyes still wander while pointing roughly at the target.
    float gaze_horizontal{0.0f};
    float gaze_vertical{0.0f};
    // Animator-driven saccade offset, summed with the target above by
    // Var::GazeH / Var::GazeV in the VM. Updated by FaceAnimator::
    // saccade_tick; kept zero when saccade is disabled.
    float gaze_saccade_h{0.0f};
    float gaze_saccade_v{0.0f};
    float eye_open_ratio{1.0f};
    float mouth_open_ratio{0.0f};
    // 0..1 loudness of the user's voice while listening (DSL var voice_level).
    float voice_level{0.0f};
    Palette palette{kDefaultPalette};
    std::uint32_t rng_state{0xC0FFEEu};
    std::optional<std::string> balloon_text{};
    // Wall-clock used for time-based animation (e.g. balloon marquee).
    std::uint32_t now_ms{0};
    // Set to `now_ms` whenever balloon_text changes — drives marquee phase.
    std::uint32_t balloon_set_ms{0};
    // Minimum display time. 0 means "use balloon defaults" (short = a fixed
    // hold, long = one marquee pass).
    std::uint32_t balloon_hold_ms{0};
    // Set by balloon rendering once the message has been displayed in full
    // (i.e. hold time elapsed for short text, or one marquee cycle for long
    // text). The render task polls this and notifies the application.
    bool balloon_done{false};
    // aora eye-ring screen coordinates (non-owning pointer to the render task's frame buffer,
    // kRingVarCount floats: l0x,l0y..l47y,r0x..r47y). With a null pointer
    // the Var::RingBase range reads 0.
    const float* aora_ring{nullptr};
};

} // namespace stackchan::avatar
