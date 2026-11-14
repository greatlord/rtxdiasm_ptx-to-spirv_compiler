cmake_minimum_required(VERSION 3.16)
project(rtxdiasm_ptx_to_spirv_compiler C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Compiler library sources (PTX -> SPIR-V), Vulkan-independent.
set(COMPILER_SOURCES
    src/control/cfg.c
    src/control/control.c
    src/control/control_flow.c
    src/host/host.c
    src/host/hoststate.c
    src/instructions/arith/add_sub.c
    src/instructions/arith/arith_convert.c
    src/instructions/arith/arith_dispatch.c
    src/instructions/arith/arith_helpers.c
    src/instructions/arith/arith_softmath.c
    src/instructions/arith/div_rem.c
    src/instructions/arith/fma.c
    src/instructions/arith/float/copysign.c
    src/instructions/arith/float/dot2.c
    src/instructions/arith/float/glsl_ops.c
    src/instructions/arith/float/minnan.c
    src/instructions/arith/float/rcp.c
    src/instructions/arith/integer/addsat.c
    src/instructions/arith/integer/bit_ops.c
    src/instructions/arith/integer/clmad.c
    src/instructions/arith/integer/dp2a.c
    src/instructions/arith/integer/dp4a.c
    src/instructions/arith/integer/mad24.c
    src/instructions/arith/integer/madwide.c
    src/instructions/arith/integer/mul24.c
    src/instructions/arith/integer/mulhi.c
    src/instructions/arith/integer/mulwide.c
    src/instructions/arith/integer/sad.c
    src/instructions/arith/madlo.c
    src/instructions/arith/minmax.c
    src/instructions/arith/mullo.c
    src/instructions/arith/unary.c
    src/instructions/compare/selp.c
    src/instructions/compare/set.c
    src/instructions/compare/setp.c
    src/instructions/compare/slct.c
    src/instructions/compare/testp.c
    src/instructions/conversion/cvt.c
    src/instructions/conversion/cvtpack.c
    src/instructions/conversion/cvtpack8.c
    src/instructions/conversion/cvt_micro.c
    src/instructions/conversion/fp8_cvt.c
    src/instructions/conversion/rnd.c
    src/instructions/data_movement/prmt.c
    src/instructions/group/barrier.c
    src/instructions/group/group.c
    src/instructions/group/isspacep.c
    src/instructions/group/ldmatrix.c
    src/instructions/group/match.c
    src/instructions/group/mbarrier.c
    src/instructions/group/mma.c
    src/instructions/group/redux.c
    src/instructions/group/shfl.c
    src/instructions/group/vote.c
    src/instructions/group/wmma.c
    src/instructions/logic_shift/logic_shift.c
    src/instructions/memory/atom.c
    src/instructions/memory/cp_async.c
    src/instructions/memory/dyn_addr.c
    src/instructions/memory/fold.c
    src/instructions/memory/ld.c
    src/instructions/memory/mem.c
    src/instructions/memory/prefetch_hint.c
    src/instructions/memory/red_async.c
    src/instructions/memory/st.c
    src/instructions/surface/surface.c
    src/instructions/texture/texture.c
    src/instructions/video/video_ops.c
    src/module/module.c
    src/module/params.c
    src/parse/compile.c
    src/parse/hostcall.c
    src/parse/lex.c
    src/parse/parse.c
    src/parse/parse_atom.c
    src/parse/parse_cvt_micro.c
    src/parse/parse_decl.c
    src/parse/parse_entry.c
    src/parse/parse_inline.c
    src/parse/parse_instr.c
    src/parse/parse_matrix.c
    src/parse/parse_mbarrier.c
    src/parse/parse_red_async.c
    src/parse/parse_surface.c
    src/parse/parse_texture.c
    src/spirv/caps.c
    src/spirv/core.c
    src/spirv/instr.c
    src/spirv/module.c
    src/spirv/values_convert.c
    src/spirv/values_mem.c
    src/spirv/values_ops.c
    src/spirv/values_resolve.c
)

function(apply_compiler_includes tgt visibility)
    target_include_directories(${tgt} ${visibility}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )
endfunction()

# Static library
add_library(rtxdiasm_ptx_to_spirv_compiler_static STATIC ${COMPILER_SOURCES})
apply_compiler_includes(rtxdiasm_ptx_to_spirv_compiler_static PUBLIC)
set_target_properties(rtxdiasm_ptx_to_spirv_compiler_static PROPERTIES
    OUTPUT_NAME rtxdiasm_ptx_to_spirv_compiler
    POSITION_INDEPENDENT_CODE ON
)

# Dynamic/shared library
add_library(rtxdiasm_ptx_to_spirv_compiler_shared SHARED ${COMPILER_SOURCES})
apply_compiler_includes(rtxdiasm_ptx_to_spirv_compiler_shared PUBLIC)
set_target_properties(rtxdiasm_ptx_to_spirv_compiler_shared PROPERTIES
    OUTPUT_NAME rtxdiasm_ptx_to_spirv_compiler
)
if(WIN32)
    # Export all symbols for simplicity unless decorated API is added later.
    set_target_properties(rtxdiasm_ptx_to_spirv_compiler_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Executable
add_executable(rtxdiasm_ptx-to-spirv_compiler cli/main.c)
target_link_libraries(rtxdiasm_ptx-to-spirv_compiler PRIVATE rtxdiasm_ptx_to_spirv_compiler_static)
target_include_directories(rtxdiasm_ptx-to-spirv_compiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Convenience alias
add_library(rtxdiasm_ptx_to_spirv_compiler ALIAS rtxdiasm_ptx_to_spirv_compiler_static)

if(MSVC)
    target_compile_options(rtxdiasm_ptx_to_spirv_compiler_static PRIVATE /W4)
    target_compile_options(rtxdiasm_ptx_to_spirv_compiler_shared PRIVATE /W4)
    target_compile_options(rtxdiasm_ptx-to-spirv_compiler PRIVATE /W4)
else()
    target_compile_options(rtxdiasm_ptx_to_spirv_compiler_static PRIVATE -Wall -Wextra)
    target_compile_options(rtxdiasm_ptx_to_spirv_compiler_shared PRIVATE -Wall -Wextra)
    target_compile_options(rtxdiasm_ptx-to-spirv_compiler PRIVATE -Wall -Wextra)
endif()
