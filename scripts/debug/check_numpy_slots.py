"""numpy 2.x의 _ARRAY_API 슬롯 테이블에서 pybind11 2.9.1이 사용하는 슬롯 번호들이
유효한지 확인한다. NULL 슬롯 → pybind11이 호출 시 SIGSEGV."""
import ctypes
import faulthandler
faulthandler.enable()

import numpy as np
print(f"numpy version: {np.__version__}")

import numpy.core.multiarray as m
api_capsule = m._ARRAY_API

# PyCapsule에서 void** 포인터 추출
import ctypes.util
lib = ctypes.CDLL(None)
_PyCapsule_GetPointer = lib.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

raw_ptr = _PyCapsule_GetPointer(api_capsule, None)
print(f"_ARRAY_API capsule raw ptr: {hex(raw_ptr)}")

# void* 배열로 접근
api_arr = (ctypes.c_void_p * 400).from_address(raw_ptr)

# pybind11 2.9.1이 사용하는 슬롯 번호들
slots = {
    211: "PyArray_GetNDArrayCFeatureVersion",
    2:   "PyArray_Type",
    3:   "PyArrayDescr_Type",
    39:  "PyVoidArrType_Type",
    45:  "PyArray_DescrFromType",
    57:  "PyArray_DescrFromScalar",
    69:  "PyArray_FromAny",
    80:  "PyArray_Resize",
    82:  "PyArray_CopyInto",
    85:  "PyArray_NewCopy",
    94:  "PyArray_NewFromDescr",
    96:  "PyArray_DescrNewFromType",
    135: "PyArray_Newshape",
    136: "PyArray_Squeeze",
    137: "PyArray_View",
    174: "PyArray_DescrConverter",
    182: "PyArray_EquivTypes",
    278: "PyArray_GetArrayParamsFromObject",
    282: "PyArray_SetBaseObject",
}

print("\n슬롯 점검 결과:")
null_slots = []
for slot_num, name in sorted(slots.items()):
    val = api_arr[slot_num]
    status = "NULL ← CRASH" if val is None or val == 0 else f"0x{val:016x}"
    print(f"  slot[{slot_num:3d}] {name:45s} = {status}")
    if val is None or val == 0:
        null_slots.append((slot_num, name))

print()
if null_slots:
    print(f"[DIAGNOSIS] NULL 슬롯 {len(null_slots)}개 발견 — pybind11 2.9.1의 슬롯 번호가")
    print(f"  numpy {np.__version__} 슬롯 테이블과 불일치. numpy 2.x에서 C API 슬롯이 재편됨.")
    for s, n in null_slots:
        print(f"  슬롯 {s} ({n}) → NULL")
else:
    print("[OK] NULL 슬롯 없음 — 슬롯 번호 불일치는 없는 상태.")
