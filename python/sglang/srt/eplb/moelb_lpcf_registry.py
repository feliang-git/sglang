# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Global per-layer registry for LP-CF runtimes.

Populated by ``ModelRunner._init_lpcf_runtimes`` when
``--ep-dispatch-algorithm=lpcf`` is enabled; looked up by
``ExpertLocationDispatchInfo.init_new`` so the per-layer TopK hook can
dispatch through the moe_load_balancer SDK without threading a runtime
argument through every call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from moe_load_balancer.adapters.sglang.lpcf.runtime import LPCFRuntime

_GLOBAL_LPCF_RUNTIMES: dict[int, "LPCFRuntime"] = {}


def get_global_lpcf_runtime(layer_id: int) -> Optional["LPCFRuntime"]:
    """Return the registered LP-CF runtime for ``layer_id`` or ``None``."""
    return _GLOBAL_LPCF_RUNTIMES.get(layer_id)


def set_global_lpcf_runtime(layer_id: int, runtime: "LPCFRuntime") -> None:
    """Register a runtime for ``layer_id``. Called once per layer at startup,
    and re-called after EPLB rebalances if the placement adapter is rebuilt.
    """
    _GLOBAL_LPCF_RUNTIMES[layer_id] = runtime


def clear_global_lpcf_runtimes() -> None:
    """Clear every registered runtime. Used by full re-init paths."""
    _GLOBAL_LPCF_RUNTIMES.clear()
