// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdio>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <stdexcept>
#include <sys/wait.h>
#include <filesystem>
#include <iostream>
#include "debug_tools_fixture.hpp"
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <variant>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/utils.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
    // Some machines will run this test on different virtual cores, so wildcard the exact coordinates.
    const std::string golden_output =
        R"(DPRINT server timed out on Device ?, worker core (x=?,y=?), riscv 4, waiting on a RAISE signal: 1)";

    void RunTest(DPrintFixture* fixture, IDevice* device) {
        // Set up program
        Program program = Program();

        // Run a kernel that just waits on a signal that never comes (BRISC only).
        constexpr CoreCoord core = {0, 0}; // Print on first core only
        KernelHandle brisc_print_kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );

        // Run the program, we expect it to throw on waiting for CQ to finish
    try {
        fixture->RunProgram(device, program);
    } catch (std::runtime_error& e) {
        const std::string expected = "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.";
        const std::string error = std::string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != std::string::npos);
    }

    // Print the actual file contents
    std::cout << "=== ACTUAL FILE CONTENTS ===" << std::endl;
    std::ifstream actual_file(DPrintFixture::dprint_file_name);
    if (actual_file.is_open()) {
        std::cout << actual_file.rdbuf();
        actual_file.close();
    } else {
        std::cout << "Could not open file: " << DPrintFixture::dprint_file_name << std::endl;
    }

    // Print the golden output
    std::cout << "=== GOLDEN OUTPUT ===" << std::endl;
    std::cout << golden_output << std::endl;

        std::cout << "ABOUT TO ASSERT\n" << std::endl;

        // Check the print log against golden output.
        ASSERT_TRUE(
            FilesMatchesString(
                DPrintFixture::dprint_file_name,
                golden_output
            )
        );
        std::cout << "THE TEST IS SUPPOSED TO HAVE PASSED AS THE ASSERT STATEMENT FINISHED\n" << std::endl;
    }
}
}


// Helper to get the directory of the currently running executable
std::string get_executable_dir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) throw std::runtime_error("Failed to read /proc/self/exe");
    result[count] = '\0';
    char* dirc = strdup(result);
    std::string dir = dirname(dirc);
    free(dirc);
    return dir;
}

// Helper to get TT_METAL_HOME from the environment
std::string get_tt_metal_home() {
    const char* env = std::getenv("TT_METAL_HOME");
    if (!env) throw std::runtime_error("TT_METAL_HOME is not set");
    return std::string(env);
}

// Helper to find watcher_dump executable robustly
std::string find_watcher_dump(const std::string& tools_dir) {
    namespace fs = std::filesystem;
    fs::path tools_path(tools_dir);

    // Check directly in tools/
    fs::path candidate = tools_path / "watcher_dump";
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
        return candidate.string();
    }

    // Check in immediate subdirectories (e.g., RelWithDebInfo, Debug, Release)
    for (const auto& entry : fs::directory_iterator(tools_path)) {
        if (entry.is_directory()) {
            fs::path sub_candidate = entry.path() / "watcher_dump";
            if (fs::exists(sub_candidate) && fs::is_regular_file(sub_candidate)) {
                return sub_candidate.string();
            }
        }
    }

    throw std::runtime_error("Could not find watcher_dump in " + tools_dir + " or its immediate subdirectories.");
}



TEST_F(DPrintFixture, TensixTestPrintHanging) {
    if (this->slow_dispatch_)
        GTEST_SKIP();

    setenv("TT_METAL_WATCHER_KEEP_ERRORS", "1", 1);

    this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, this->devices_[0]);

    // Find watcher_dump executable
    std::string watcher_dump_path = find_watcher_dump(std::string(BUILD_ROOT_DIR) + "/tools");

    // Run watcher_dump tool using filesystem
    std::string command = watcher_dump_path + " -d=0 -w -c";
    int result = std::system(command.c_str());
    ASSERT_EQ(result, 0) << "watcher_dump failed with exit code: " << result;

    std::cout << "we managed to get past the watcher_dump tool" << std::endl;

    // Check watcher log file
    std::string watcher_log_path = "generated/watcher/watcher.log";
    std::ifstream watcher_log(watcher_log_path);
    ASSERT_TRUE(watcher_log.is_open()) << "Failed to open watcher log: " << watcher_log_path;

    std::string line;
    bool found = false;
    const std::string expected_str = "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp";
    while (std::getline(watcher_log, line)) {
        if (line.find(expected_str) != std::string::npos) {
            found = true;
            break;
        }
    }
    watcher_log.close();
    ASSERT_TRUE(found) << "Error: couldn't find expected string in watcher log after dump: " << expected_str;

    std::cout << "Tearing down and reinitializing" << std::endl;
    this->ShutdownAllCores();

    MetalContext::instance().reinitialize();

    // Clean up
    //std::filesystem::remove(watcher_log_path);
}
