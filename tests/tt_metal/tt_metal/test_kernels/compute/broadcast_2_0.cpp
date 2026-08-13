// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"

#ifndef BCAST_ROW_IDX
#define BCAST_ROW_IDX 0
#endif

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t icb0 = dfb::in0;
    constexpr uint32_t icb1 = dfb::in1;
    constexpr uint32_t ocb = dfb::out;

#ifndef BCAST_OP_INIT
    compute_kernel_hw_startup(icb0, icb1, ocb);
    bcast_init<BCAST_LLKOP, BCAST_DIM>(icb0, icb1);
#else
    compute_kernel_hw_startup(icb0, icb1, ocb);
    // This test deliberately exercises the deprecated *_bcast_*_init_short shims (tracked in
    // .github/deprecations.json, removal after September 15th, 2026) to cover the legacy API
    // convention. Suppress the deprecation warning locally so it does not clobber CI logs.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    BCAST_OP_INIT(icb0, icb1);
#pragma GCC diagnostic pop
#endif

    dfb1.wait_front(onetile);
    dfb_out.reserve_back(onetile);
    tile_regs_acquire();
    tile_regs_wait();
    dfb0.wait_front(onetile);

#ifndef BCAST_SPECIFIC
    // For template version, use compile-time check for ROW broadcast
    if constexpr (BCAST_DIM == BroadcastType::ROW) {
        BCAST_OP<BCAST_DIM>(icb0, icb1, 0, 0, 0, BCAST_ROW_IDX);
    } else {
        BCAST_OP<BCAST_DIM>(icb0, icb1, 0, 0, 0);
    }
#else
// For specific function calls, check if BCAST_IS_ROW is defined
#ifdef BCAST_IS_ROW
    // Row broadcast functions have the bcast_row_idx parameter
    BCAST_OP(icb0, icb1, 0, 0, 0, BCAST_ROW_IDX);
#else
    // Col and Scalar broadcast functions don't have the parameter
    BCAST_OP(icb0, icb1, 0, 0, 0);
#endif
#endif

    pack_tile(0, ocb);

    dfb0.pop_front(onetile);
    tile_regs_commit();
    tile_regs_release();
    dfb_out.push_back(onetile);
    dfb1.pop_front(onetile);
}
