# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from typing import Any


class Tracer(Callable):
    """Wrapper for capturing and executing a trace of a given function."""

    def __init__(self, function: Callable[..., Any], /, *, device: ttnn.MeshDevice) -> None:
        """Initialize the tracer.

        Args:
            function: Function to be traced.
            device: Device on which to capture and execute the trace.
        """
        self._function = function
        self._device = device
        self._inputs: dict[str, Any] = {}
        self._outputs: Any = None
        self._trace_id: ttnn.MeshTraceId | None = None

    def set_inputs(self, **kwargs: Any) -> None:
        """Set or update named inputs.

        Before capture, inputs are validated and copied to `device` if needed. After capture, only
        previously-declared names may be updated; non-tensors must equal the initial value.
        """
        for name, value in kwargs.items():
            if self._trace_id is None:
                self._inputs[name] = self._prepare_initial_input(name, value)
            else:
                self._update_input(name, value)

    def __call__(self, *, tracer_cq_id: int = 0, tracer_blocking_execution: bool = True, **kwargs: Any) -> Any:
        """Capture or execute trace.

        On the first call, runs the wrapped function twice, once to compile, and once to capture the
        trace. On subsequent calls, executes the captured trace.

        Args:
            tracer_cq_id: Command queue id.
            tracer_blocking_execution: Whether `ttnn.execute_trace` should block.
            **kwargs: Named inputs to set before invocation. This is equivalent to calling
                `set_inputs(**kwargs)` before `run()`.

        Returns:
            The outputs of the wrapped function.

        Raises:
            TypeError: If outputs have unsupported types.
            Any exception raised by the wrapped function during first invocation.
        """
        self.set_inputs(**kwargs)

        if self._trace_id is None:
            # compile
            self._function(**self._inputs)

            # capture trace
            trace_id = ttnn.begin_trace_capture(self._device, cq_id=tracer_cq_id)
            try:
                try:
                    outputs = self._function(**self._inputs)
                finally:
                    ttnn.end_trace_capture(self._device, trace_id, cq_id=tracer_cq_id)

                self._check_outputs(outputs)
            except Exception:
                ttnn.release_trace(self._device, trace_id)
                raise

            self._trace_id = trace_id
            self._outputs = outputs
        else:
            ttnn.execute_trace(self._device, self._trace_id, cq_id=tracer_cq_id, blocking=tracer_blocking_execution)

        return self._outputs

    def release(self) -> None:
        """Release the captured trace and clear inputs and outputs."""
        trace_id = self._trace_id

        if trace_id is not None:
            self._trace_id = None
            self._inputs = {}
            self._outputs = None
            ttnn.release_trace(self._device, trace_id)

    def _prepare_initial_input(self, name: str, value: Any) -> Any:
        """Validate input and move to device if needed."""
        if isinstance(value, ttnn.Tensor):
            if value.device() is None:
                return value.to(self._device)
            if value.device() == self._device:
                return value
            msg = f"input '{name}' device {value.device()} does not match tracer device {self._device}"
            raise ValueError(msg)

        if isinstance(value, Hashable):
            return value

        # inputs that are not tensors should be immutable, so we restrict to hashable types, which
        # usually are
        msg = f"input '{name}' should be a ttnn.Tensor or hashable type, got {type(value)}"
        raise TypeError(msg)

    def _update_input(self, name: str, value: Any) -> None:
        """Update an input after capture, enforcing shape/type/layout/device."""
        if name not in self._inputs:
            msg = f"input '{name}' was not in the initial inputs"
            raise KeyError(msg)

        orig_value = self._inputs[name]

        if type(value) is not type(orig_value):
            msg = f"input '{name}' type {type(value)} does not match the initial type {type(orig_value)}"
            raise TypeError(msg)

        if isinstance(value, ttnn.Tensor):
            if value.shape != orig_value.shape or value.dtype != orig_value.dtype or value.layout != orig_value.layout:
                msg = f"input '{name}' tensor properties do not match the initial value"
                raise ValueError(msg)

            if value.device() is None:
                ttnn.copy_host_to_device_tensor(value, orig_value)
            else:
                if value.device() != self._device:
                    msg = f"input '{name}' device {value.device()} does not match tracer device {self._device}"
                    raise ValueError(msg)

                ttnn.copy(value, orig_value)
        elif value != orig_value:
            msg = f"input '{name}' does not match the initial value"
            raise ValueError(msg)

    def _check_outputs(self, outputs: Any) -> None:
        """Ensure outputs are nested structures of Tensors/Hashables with str keys."""
        # since we store outputs for later use, they should be tensors or immutable, which hashable
        # types usually are
        if isinstance(outputs, (tuple, list)):
            for x in outputs:
                self._check_outputs(x)
        elif isinstance(outputs, dict):
            for k, v in outputs.items():
                if not isinstance(k, str):
                    msg = f"output dict has non-str key of type {type(k)}"
                    raise TypeError(msg)
                self._check_outputs(v)
        elif isinstance(outputs, (ttnn.Tensor, Hashable)):
            pass
        else:
            msg = f"output has unsupported type {type(outputs)}"
            raise TypeError(msg)
