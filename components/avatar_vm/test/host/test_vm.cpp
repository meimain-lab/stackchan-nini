// SPDX-FileCopyrightText: 2026 Kenta IDA <fuga@fugafuga.org>
// SPDX-License-Identifier: BSL-1.0
//
// Execution / boundary tests for avatar_vm::Vm::run(). Confirms the
// interpreter both (a) computes the right values through the arithmetic +
// control-flow opcodes and emits the expected draw calls, and (b) rejects
// out-of-range operands (bad const/var/fn/local ids, stack under/overflow,
// call-depth blowout, divide-by-zero, out-of-bounds jumps) with the matching
// VmError instead of misbehaving.

#include <cstdint>
#include <span>
#include <vector>

#include "avatar_vm/bytecode.hpp"
#include "avatar_vm/opcodes.hpp"
#include "avatar_vm/vm.hpp"
#include "test_support.hpp"

using namespace stackchan::avatar_vm;
using avtest::BytecodeBuilder;
using avtest::DrawOp;
using avtest::RecordingCanvas;

namespace {

// Opcode byte shorthands.
constexpr std::uint8_t PUSH_I8 = 0x02;
constexpr std::uint8_t PUSH_I16 = 0x03;
constexpr std::uint8_t PUSH_CONST = 0x04;
constexpr std::uint8_t PUSH_VAR = 0x05;
constexpr std::uint8_t PUSH_LOCAL = 0x06;
constexpr std::uint8_t STORE_LOCAL = 0x07;
constexpr std::uint8_t POP = 0x08;
constexpr std::uint8_t DUP = 0x09;
constexpr std::uint8_t ADD = 0x10;
constexpr std::uint8_t MUL = 0x12;
constexpr std::uint8_t DIV = 0x13;
constexpr std::uint8_t JMP = 0x30;
constexpr std::uint8_t CALL = 0x33;
constexpr std::uint8_t RET = 0x34;
constexpr std::uint8_t FILL_RECT = 0x40;
constexpr std::uint8_t FILL_CIRCLE = 0x41;

// Run a decoded program against a default 320x240 canvas + neutral context.
struct RunResult {
    tl::expected<Bytecode, VmError> decoded;
    tl::expected<void, VmError> ran;
    RecordingCanvas canvas{320, 240};
};

RunResult run_program(const std::vector<std::uint8_t>& buf, stackchan::avatar::DrawContext ctx = {}) {
    RunResult rr;
    rr.decoded = decode(std::span<const std::uint8_t>(buf));
    if (!rr.decoded) {
        rr.ran = tl::unexpected(rr.decoded.error());
        return rr;
    }
    stackchan::avatar::FaceTuning tuning;
    Vm vm;
    rr.ran = vm.run(*rr.decoded, rr.canvas, ctx, tuning);
    return rr;
}

} // namespace

int main() {
    // --- entry returns immediately ---------------------------------------
    {
        BytecodeBuilder b;
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.empty());
    }

    // --- arithmetic then FillRect emits one op with correct color --------
    {
        BytecodeBuilder b;
        // push x=10 y=20 w=30 h=40 color=0x1234, fillRect, ret
        b.code(PUSH_I8);
        b.code(10);
        b.code(PUSH_I8);
        b.code(20);
        b.code(PUSH_I8);
        b.code(30);
        b.code(PUSH_I8);
        b.code(40);
        b.code(PUSH_I16);
        b.code_i16(static_cast<std::int16_t>(0x1234));
        b.code(FILL_RECT);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            const auto& op = rr.canvas.ops[0];
            CHECK(op.kind == DrawOp::Kind::FillRect);
            CHECK(op.a == 10 && op.b == 20 && op.c == 30 && op.d == 40);
            CHECK(op.color == 0x1234);
        }
    }

    // --- Add computes a+b -------------------------------------------------
    {
        BytecodeBuilder b;
        // fillCircle(cx=(3+4)=7, cy=5, r=2, color=9)
        b.code(PUSH_I8);
        b.code(3);
        b.code(PUSH_I8);
        b.code(4);
        b.code(ADD);
        b.code(PUSH_I8);
        b.code(5);
        b.code(PUSH_I8);
        b.code(2);
        b.code(PUSH_I8);
        b.code(9);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].kind == DrawOp::Kind::FillCircle);
            CHECK(rr.canvas.ops[0].a == 7);
        }
    }

    // --- locals: store then load round-trips -----------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_I8);
        b.code(42);
        b.code(STORE_LOCAL);
        b.code(0);
        b.code(PUSH_LOCAL);
        b.code(0);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE); // cx=42 cy=1 r=1 color=3
        b.code(RET);
        b.add_fn(/*code_offset=*/0, /*param_count=*/0, /*local_count=*/1);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1 && rr.canvas.ops[0].a == 42);
    }

    // --- PushConst references the const table ----------------------------
    {
        BytecodeBuilder b;
        b.add_const_i32(77);
        b.code(PUSH_CONST);
        b.code(0);
        b.code(POP);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
    }

    // --- PushVar CanvasW reads the canvas width --------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(0x00); // Var::CanvasW
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1 && rr.canvas.ops[0].a == 320);
    }

    // --- PushVar Expr / ExprFrom / ExprBlend read DrawContext ------------
    {
        BytecodeBuilder b;
        // fillCircle(cx=expr, cy=expr_from, r=expr_blend*10, color=3)
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::Expr));
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprFrom));
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprBlend));
        b.code(PUSH_I8);
        b.code(10);
        b.code(MUL);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);

        stackchan::avatar::DrawContext ctx;
        ctx.expression = stackchan::avatar::Expression::Happy;    // 1
        ctx.expression_from = stackchan::avatar::Expression::Sad; // 2
        ctx.expression_blend = 0.5f;
        auto rr = run_program(b.build(0), ctx);
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].kind == DrawOp::Kind::FillCircle);
            CHECK(rr.canvas.ops[0].a == 1);
            CHECK(rr.canvas.ops[0].b == 2);
            CHECK(rr.canvas.ops[0].c == 5); // 0.5 * 10
        }
    }

    // --- default ctx: expr_from Neutral, expr_blend idle (=1) ------------
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprFrom));
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprBlend));
        b.code(PUSH_I8);
        b.code(10);
        b.code(MUL);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].a == 0);  // Neutral
            CHECK(rr.canvas.ops[0].b == 10); // 1.0 * 10
        }
    }

    // --- PushVar ExprHoldTo / ExprHoldBlend ------------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprHoldTo));
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprHoldBlend));
        b.code(PUSH_I8);
        b.code(10);
        b.code(MUL);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);

        stackchan::avatar::DrawContext ctx;
        ctx.expression_hold_to = stackchan::avatar::Expression::Angry; // 3
        ctx.expression_hold_blend = 0.25f;
        auto rr = run_program(b.build(0), ctx);
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].a == 3);
            CHECK(rr.canvas.ops[0].b == 2); // 0.25 * 10
        }
    }

    // --- PushVar ExprHold2To / ExprHold2Blend ----------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprHold2To));
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::ExprHold2Blend));
        b.code(PUSH_I8);
        b.code(10);
        b.code(MUL);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);

        stackchan::avatar::DrawContext ctx;
        ctx.expression_hold2_to = stackchan::avatar::Expression::Happy; // 1
        ctx.expression_hold2_blend = 0.5f;
        auto rr = run_program(b.build(0), ctx);
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].a == 1);
            CHECK(rr.canvas.ops[0].b == 5); // 0.5 * 10
        }
    }

    // --- function call with a parameter ----------------------------------
    {
        BytecodeBuilder b;
        // fn0 (entry): push 5, call fn1, ret
        // fn1 (1 param): fillCircle(param, 1, 1, 2), ret
        b.code(PUSH_I8); // 0
        b.code(5);       // 1
        b.code(CALL);    // 2
        b.code(1);       // 3 fn_id=1
        b.code(RET);     // 4
        const std::size_t fn1_off = b.code_len();
        b.code(PUSH_LOCAL); // param 0
        b.code(0);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(2);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);
        b.add_fn(static_cast<std::uint16_t>(fn1_off), /*param_count=*/1, /*local_count=*/1);
        auto rr = run_program(b.build(0));
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1 && rr.canvas.ops[0].a == 5);
    }

    // ================= error / boundary cases ============================

    // --- stack underflow (Add with empty stack) --------------------------
    {
        BytecodeBuilder b;
        b.code(ADD);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::StackUnderflow);
    }

    // --- stack overflow (push in a tight loop past kStackSize=64) --------
    {
        BytecodeBuilder b;
        // loop: dup pushes; start with one value, Dup repeatedly, Jmp back.
        b.code(PUSH_I8);
        b.code(1);
        const std::size_t loop = b.code_len();
        b.code(DUP);
        b.code(JMP);
        // jump back to `loop`: offset from pc-after-operand to loop.
        // We'll patch after we know the size; compute manually:
        // after this Jmp operand, pc = loop_start + 1(DUP) + 3(JMP w/ operand).
        {
            std::int16_t pc_after = static_cast<std::int16_t>(loop + 1 + 3);
            std::int16_t off = static_cast<std::int16_t>(static_cast<std::int16_t>(loop) - pc_after);
            b.code_i16(off);
        }
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::StackOverflow);
    }

    // --- divide by zero --------------------------------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(0);
        b.code(DIV);
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::DivideByZero);
    }

    // --- bad const id ----------------------------------------------------
    {
        BytecodeBuilder b;
        b.add_const_i32(1);
        b.code(PUSH_CONST);
        b.code(9); // only const 0 exists
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::BadConstId);
    }

    // --- voice_level (first id after the ring range) reads DrawContext ----
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(static_cast<std::uint8_t>(Var::VoiceLevel));
        b.code(PUSH_I8);
        b.code(20);
        b.code(MUL);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(1);
        b.code(PUSH_I8);
        b.code(3);
        b.code(FILL_CIRCLE);
        b.code(RET);
        b.add_fn(0, 0, 0);

        stackchan::avatar::DrawContext ctx;
        ctx.voice_level = 0.5f;
        auto rr = run_program(b.build(0), ctx);
        CHECK(rr.ran.has_value());
        CHECK(rr.canvas.ops.size() == 1);
        if (rr.canvas.ops.size() == 1) {
            CHECK(rr.canvas.ops[0].a == 10); // 0.5 * 20
        }
    }

    // --- bad var id ------------------------------------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_VAR);
        b.code(0xFE); // >= Var::RingBase + kRingVarCount
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::BadVarId);
    }

    // --- bad local slot --------------------------------------------------
    {
        BytecodeBuilder b;
        b.code(PUSH_LOCAL);
        b.code(5); // fn declares 0 locals
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::BadLocalSlot);
    }

    // --- bad fn id on Call -----------------------------------------------
    {
        BytecodeBuilder b;
        b.code(CALL);
        b.code(9); // only fn 0
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::BadFnId);
    }

    // --- call depth exceeded (fn calls itself forever) -------------------
    {
        BytecodeBuilder b;
        // entry fn0: ret (never reached — we make fn1 recurse)
        // Actually: entry fn0 calls fn1; fn1 calls fn1.
        b.code(CALL); // 0
        b.code(1);    // 1
        b.code(RET);  // 2
        const std::size_t fn1_off = b.code_len();
        b.code(CALL);
        b.code(1); // recurse into itself
        b.code(RET);
        b.add_fn(0, 0, 0);
        b.add_fn(static_cast<std::uint16_t>(fn1_off), 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::CallDepthExceeded);
    }

    // --- jump out of bounds ----------------------------------------------
    {
        BytecodeBuilder b;
        b.code(JMP);
        b.code_i16(1000); // far past the end
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::JumpOutOfBounds);
    }

    // --- unknown opcode --------------------------------------------------
    {
        BytecodeBuilder b;
        b.code(0xEE); // not a defined Op
        b.code(RET);
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::UnknownOpcode);
    }

    // --- truncated immediate (PushI16 with only 1 operand byte) ----------
    {
        BytecodeBuilder b;
        b.code(PUSH_I16);
        b.code(0x00); // missing second byte; code ends here
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::Truncated);
    }

    // --- running off the end of code (no Ret) ----------------------------
    {
        BytecodeBuilder b;
        b.code(0x00); // Nop, then code ends without Ret
        b.add_fn(0, 0, 0);
        auto rr = run_program(b.build(0));
        CHECK(!rr.ran && rr.ran.error() == VmError::Truncated);
    }

    // Expression and ExprValue are the same u8 protocol. 0-5 stay the
    // original six; the extended expressions occupy 6-12. Idle is an alias
    // of Neutral.
    {
        using stackchan::avatar::Expression;
        CHECK(static_cast<int>(Expression::Neutral) == 0);
        CHECK(static_cast<int>(Expression::Happy) == 1);
        CHECK(static_cast<int>(Expression::Sad) == 2);
        CHECK(static_cast<int>(Expression::Angry) == 3);
        CHECK(static_cast<int>(Expression::Doubt) == 4);
        CHECK(static_cast<int>(Expression::Sleepy) == 5);
        CHECK(static_cast<int>(Expression::Listening) == 6);
        CHECK(static_cast<int>(Expression::Thinking) == 7);
        CHECK(static_cast<int>(Expression::Excited) == 8);
        CHECK(static_cast<int>(Expression::Curious) == 9);
        CHECK(static_cast<int>(Expression::Confused) == 10);
        CHECK(static_cast<int>(Expression::Surprised) == 11);
        CHECK(static_cast<int>(Expression::Dizzy) == 12);
        CHECK(static_cast<int>(Expression::Affection) == 13);
        CHECK(static_cast<int>(ExprValue::Neutral) == static_cast<int>(Expression::Neutral));
        CHECK(static_cast<int>(ExprValue::Happy) == static_cast<int>(Expression::Happy));
        CHECK(static_cast<int>(ExprValue::Sad) == static_cast<int>(Expression::Sad));
        CHECK(static_cast<int>(ExprValue::Angry) == static_cast<int>(Expression::Angry));
        CHECK(static_cast<int>(ExprValue::Doubt) == static_cast<int>(Expression::Doubt));
        CHECK(static_cast<int>(ExprValue::Sleepy) == static_cast<int>(Expression::Sleepy));
        CHECK(static_cast<int>(ExprValue::Listening) == static_cast<int>(Expression::Listening));
        CHECK(static_cast<int>(ExprValue::Thinking) == static_cast<int>(Expression::Thinking));
        CHECK(static_cast<int>(ExprValue::Excited) == static_cast<int>(Expression::Excited));
        CHECK(static_cast<int>(ExprValue::Curious) == static_cast<int>(Expression::Curious));
        CHECK(static_cast<int>(ExprValue::Confused) == static_cast<int>(Expression::Confused));
        CHECK(static_cast<int>(ExprValue::Surprised) == static_cast<int>(Expression::Surprised));
        CHECK(static_cast<int>(ExprValue::Dizzy) == static_cast<int>(Expression::Dizzy));
        CHECK(static_cast<int>(ExprValue::Affection) == static_cast<int>(Expression::Affection));
    }

    return avtest::finish("vm");
}
