#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>

typedef enum {
    LOG_INFO,
    LOG_DEBUG,
    LOG_WARNING,
    LOG_ERROR,
    LOG_FATAL,
    LOG_MODULE,
    LOG_RESET,
    NUMBER_LOGS
} LogColor;

#define COLOR_RESET    "\x1b[0m"
#define COLOR_INFO     "\x1b[35m"   // Cyan
#define COLOR_DEBUG    "\x1b[36m"   // White
#define COLOR_WARNING  "\x1b[33m"   // Yellow
#define COLOR_ERROR    "\x1b[31m"   // Red
#define COLOR_FATAL    "\x1b[31;5m" // Flashing Red
#define COLOR_MODULE   "\x1b[90m"   // Dark Gray for metadata
#define COLOR_THREAD   "\x1b[34m"   // Blue for identifiers
#define COLOR_TAG      "\x1b[35m"   // Magenta for custom string tags

#define stdlog stderr

typedef struct {
    const char *color;
    LogColor level;
    const char *name;
} Log;

static const Log logs[NUMBER_LOGS] = {
    [LOG_INFO]    = { .color = COLOR_INFO,    .level = LOG_INFO,    .name = "INFO" },
    [LOG_DEBUG]   = { .color = COLOR_DEBUG,   .level = LOG_DEBUG,   .name = "DEBUG" },
    [LOG_WARNING] = { .color = COLOR_WARNING, .level = LOG_WARNING, .name = "WARN" },
    [LOG_ERROR]   = { .color = COLOR_ERROR,   .level = LOG_ERROR,   .name = "ERROR" },
    [LOG_FATAL]   = { .color = COLOR_FATAL,   .level = LOG_FATAL,   .name = "FATAL" },
    [LOG_MODULE]  = { .color = COLOR_MODULE,  .level = LOG_MODULE,  .name = "MOD" },
    [LOG_RESET]   = { .color = COLOR_RESET,   .level = LOG_RESET,   .name = "RESET" }
};

static inline const char* _log_get_filename(const char* filepath) {
    const char *filename = strrchr(filepath, '/');
    #ifdef _WIN32
    if (!filename) filename = strrchr(filepath, '\\');
    #endif
    return filename ? filename + 1 : filepath;
}

static inline void _impl_printLog(FILE *stream, const char *file_path, int line, bool show_thread, bool show_file_line, const char *tag, int current_errno, const LogColor logLevel, const char *restrict format, ...) {
    if (logLevel < 0 || logLevel >= NUMBER_LOGS) return;

    va_list args;
    va_start(args, format);

    const char *filename = _log_get_filename(file_path);
    bool is_terminal = (stream == stdout || stream == stderr);

    flockfile(stream);

    // 1. Output Metadata Prefix Tags
    if (is_terminal) {
        if (show_thread) {
            fprintf(stream, "%s[P:%d]%s ", COLOR_THREAD, (int)getpid(), COLOR_RESET);
        }
        if (show_file_line) {
            fprintf(stream, "%s[%s:%d]%s ", COLOR_MODULE, filename, line, COLOR_RESET);
        }
        if (tag != NULL && strlen(tag) > 0) {
            fprintf(stream, "%s[%s]%s ", COLOR_TAG, tag, COLOR_RESET);
        }
        fprintf(stream, "%s[%s]%s : ", logs[logLevel].color, logs[logLevel].name, COLOR_RESET);
    } else {
        if (show_thread) {
            fprintf(stream, "[P:%d] ", (int)getpid());
        }
        if (show_file_line) {
            fprintf(stream, "[%s:%d] ", filename, line);
        }
        if (tag != NULL && strlen(tag) > 0) {
            fprintf(stream, "[%s] ", tag);
        }
        fprintf(stream, "[%s] : ", logs[logLevel].name);
    }

    // 2. Output Main User Log Body
    vfprintf(stream, format, args);
    
    // 3. Auto-Append Errno Metrics
    if ((logLevel == LOG_ERROR || logLevel == LOG_FATAL) && current_errno != 0) {
        fprintf(stream, " [Errno %d: %s]", current_errno, strerror(current_errno));
    }

    fprintf(stream, "\n");
    fflush(stream);

    funlockfile(stream);
    va_end(args);
}

// -------------------------------------------------------------------------
// COMPILER MACRO LOGIC ROUTER
// -------------------------------------------------------------------------
#ifdef NDEBUG
    #define _LOG_SELECT(show_thread, show_file_line, tag, level, fmt, ...) \
        do { if ((level) != LOG_DEBUG) { _impl_printLog(stdlog, __FILE__, __LINE__, show_thread, show_file_line, tag, errno, level, fmt, ##__VA_ARGS__); } } while (0)
#else
    #define _LOG_SELECT(show_thread, show_file_line, tag, level, fmt, ...) \
        _impl_printLog(stdlog, __FILE__, __LINE__, show_thread, show_file_line, tag, errno, level, fmt, ##__VA_ARGS__)
#endif

// Positional selection helpers to parse tag string vs log level integer
#define _LOG_GET_7TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, ...) arg7
#define _LOG_EXT_CHOOSER(...) _LOG_GET_7TH_ARG(__VA_ARGS__, _LOG_SELECT, _LOG_SELECT, _LOG_SELECT, _LOG_SELECT, _LOG_SELECT, _LOG_SELECT)

// -------------------------------------------------------------------------
// PUBLIC CONSOLE LOGGING INTERFACES
// -------------------------------------------------------------------------

// Standard fast log: Hides threads, filenames, and line numbers by default
#define LOG(level, fmt, ...) \
    _LOG_SELECT(false, false, NULL, level, fmt, ##__VA_ARGS__)

// Extended log: LOG_EXT(show_thread, show_file_line, [tag], level, fmt, ...)
#define LOG_EXT(show_thread, show_file_line, arg3, ...) \
    _LOG_EXT_CHOOSER(show_thread, show_file_line, arg3, ##__VA_ARGS__)(show_thread, show_file_line, _Generic((arg3), char*: arg3, const char*: arg3, default: NULL), _Generic((arg3), char*: __VA_ARGS__, const char*: __VA_ARGS__, default: arg3, ##__VA_ARGS__))

