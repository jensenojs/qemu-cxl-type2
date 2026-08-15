/*
 * Bounded production-path gate for the CXL Type-2 pageable model alias.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef CXL_TYPE2_MODEL_ALIAS_GATE_H
#define CXL_TYPE2_MODEL_ALIAS_GATE_H

#include "qapi/error.h"

typedef struct CXLType2State CXLType2State;

bool cxl_type2_model_alias_gate_run(CXLType2State *ct2d, Error **errp);

#endif
