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
    R"(DPRINT server timed out on Device ?, worker core (x=?,y=?), riscv 4, waiting on a RAISE signal: 1
)";

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

    // Check the print log against golden output.
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            golden_output
        )
    );
}

static void RunTest(
    WatcherFixture* fixture,
    IDevice* device,
    riscv_id_t riscv_type,
    debug_assert_type_t assert_type = DebugAssertTripped) {
    // Set up program
    Program program = Program();

    // Depending on riscv type, choose one core to run the test on (since the test hangs the board).
    CoreCoord logical_core, virtual_core;
    if (riscv_type == DebugErisc) {
        if (device->get_active_ethernet_cores(true).empty()) {
            log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_active_ethernet_cores(true).begin());
        virtual_core = device->ethernet_core_from_logical_core(logical_core);
    } else if (riscv_type == DebugIErisc) {
        if (device->get_inactive_ethernet_cores().empty()) {
            log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_inactive_ethernet_cores().begin());
        virtual_core = device->ethernet_core_from_logical_core(logical_core);
    } else {
        logical_core = CoreCoord{0, 0};
        virtual_core = device->worker_core_from_logical_core(logical_core);
    }
    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    std::string risc;
    switch(riscv_type) {
        case DebugBrisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default
                }
            );
            risc = " brisc";
            break;
        case DebugNCrisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default
                }
            );
            risc = "ncrisc";
            break;
        case DebugTrisc0:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC0", "1"}}
                }
            );
            risc = "trisc0";
            break;
        case DebugTrisc1:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC1", "1"}}
                }
            );
            risc = "trisc1";
            break;
        case DebugTrisc2:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC2", "1"}}
                }
            );
            risc = "trisc2";
            break;
        case DebugErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            risc = "erisc";
            break;
        case DebugIErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            risc = "erisc";
            break;
        default: log_info(tt::LogTest, "Unsupported risc type: {}, skipping test...", riscv_type); GTEST_SKIP();
    }

    // Write runtime args that should not trip an assert.
    const std::vector<uint32_t> safe_args = {3, 4, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program, assert_kernel, logical_core, safe_args);

    // Run the kernel, don't expect an issue here.
    log_info(LogTest, "Running args that shouldn't assert...");
    // TODO: #24887, ND issue with this test - only run once below when issue is fixed
    fixture->RunProgram(device, program);
    fixture->RunProgram(device, program);
    fixture->RunProgram(device, program);
    log_info(LogTest, "Args did not assert!");

    // Write runtime args that should trip an assert.
    const std::vector<uint32_t> unsafe_args = {3, 3, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program, assert_kernel, logical_core, unsafe_args);

    // Run the kerel, expect an exit due to the assert.
    log_info(LogTest, "Running args that should assert...");
    fixture->RunProgram(device, program);

    // We should be able to find the expected watcher error in the log as well,
    // expected error message depends on the risc we're running on and the assert type.
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";
    std::string expected;
    if (assert_type == DebugAssertTripped) {
        const uint32_t line_num = 67;
        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} tripped an assert on line {}. Current kernel: "
            "{}.",
            device->id(),
            (riscv_type == DebugErisc) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            line_num,
            kernel);
        expected +=
            " Note that file name reporting is not yet implemented, and the reported line number for the assert may be "
            "from a different file.";
    } else {
        std::string barrier;
        if (assert_type == DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped) {
            barrier = "NOC non-posted atomics flushed";
        } else if (assert_type == DebugAssertNCriscNOCNonpostedWritesSentTripped) {
            barrier = "NOC non-posted writes sent";
        } else if (assert_type == DebugAssertNCriscNOCPostedWritesSentTripped) {
            barrier = "NOC posted writes sent";
        } else if (assert_type == DebugAssertNCriscNOCReadsFlushedTripped) {
            barrier = "NOC reads flushed";
        }

        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} detected an inter-kernel data race due to "
            "kernel completing with pending NOC transactions (missing {} barrier). Current kernel: "
            "{}.",
            device->id(),
            (riscv_type == DebugErisc) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            barrier,
            kernel);
    }

    log_info(LogTest, "Expected error: {}", expected);
    std::string exception = "";
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception == "");
    log_info(LogTest, "Reported error: {}", exception);
    EXPECT_TRUE(expected == MetalContext::instance().watcher_server()->exception_message());
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

// Integration test replicating tests/scripts/run_tools_tests.sh
TEST_F(DPrintFixture, WatcherDumpPrintHanging) {
    // Compute paths
    std::string exe_dir = get_executable_dir();
    std::string watcher_dump_path = find_watcher_dump(std::string(BUILD_ROOT_DIR) + "/tools");
    std::string watcher_log_path = get_tt_metal_home() + "/generated/watcher/watcher.log";

    setenv("TT_METAL_WATCHER_KEEP_ERRORS", "1", 1);

    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice([](DPrintFixture* fixture, IDevice* device) {
        CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device);},
        this->devices_[0]);

    //Run watcher_dump tool as in the bash script: ./build/tools/watcher_dump -d=0 -w -c
    // Run watcher_dump in a completely separate process to avoid device conflicts
    pid_t pid = fork();

    if (pid == 0) {  // Child process
        // Prepare arguments
        const char* args[] = {
            watcher_dump_path.c_str(),
            "-d=0",
            "-w",
            "-c",
            nullptr
        };

        // Execute the command - this replaces the entire child process
        execv(watcher_dump_path.c_str(), const_cast<char* const*>(args));

        // If execv fails
        perror("execv failed");
        exit(1);
    } else if (pid > 0) {  // Parent process
        int status;
        pid_t result = waitpid(pid, &status, 0);

        ASSERT_TRUE(result > 0) << "Failed to wait for child process";
        ASSERT_TRUE(WIFEXITED(status)) << "Child process did not exit normally";
        ASSERT_EQ(WEXITSTATUS(status), 0) << "Child process exited with non-zero status";
    } else {
        FAIL() << "Fork failed";
    }

    printf("we managed to get past the watcher_dump tool\n");

    // Clear device maps to prevent teardown from trying to close corrupted devices
    // watcher_dump may have corrupted the device state, so we skip normal teardown
    this->ClearDeviceMaps();

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

    std::filesystem::remove(watcher_log_path);


}

TEST_F(WatcherFixture, TestWatcherAssertBrisc) {
    // Compute paths
    std::string exe_dir = get_executable_dir();
    std::string watcher_dump_path = find_watcher_dump(std::string(BUILD_ROOT_DIR) + "/tools");
    std::string watcher_log_path = get_tt_metal_home() + "/generated/watcher/watcher.log";

    // Save current directory and change to TT_METAL_HOME
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        throw std::runtime_error("Failed to get current directory");
    }
    std::string original_cwd = cwd;

    setenv("TT_METAL_WATCHER_KEEP_ERRORS", "1", 1);

    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }

    // Only run on device 0 because this test takes the watcher server down.
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device, DebugBrisc);},
        this->devices_[0]
    );

}
