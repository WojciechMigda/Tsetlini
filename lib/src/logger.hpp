// Copyright 2013 Krzysztof Jusiak
// https://gist.github.com/krzysztof-jusiak/6434935
//
// Copytight 2018 Wojciech Migda
//

#pragma once

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <sstream>

struct __logger_none_t
{
  struct init_tag{};
  explicit __logger_none_t(init_tag){} // to disable default constructor
};

enum log_level
{
    error = 1,
    warn  = 2,
    debug = 3,
    info  = 4,
    trace = 5
};

inline std::ostream& operator<<(std::ostream& p_ostream, const log_level& log)
{
    switch (log)
    {
        case error: p_ostream << "ERROR";   break;
        case warn:  p_ostream << "WARNING";    break;
        case debug: p_ostream << "DEBUG";   break;
        case info:  p_ostream << "INFO";    break;
        case trace: p_ostream << "TRACE";   break;
        default:    p_ostream << "UNKNOWN"; break;
    }

    return p_ostream;
}

namespace logger
{

template<typename log_type, typename log_level, log_level log, unsigned level>
class logger_impl
{
public:
    logger_impl(const std::string & file, unsigned line, const std::string & function)
    {
        if (log <= level)
        {
            stream_ << file << ":" << line << ":0: " << log << ": " << function << ": ";
        }
    }

    logger_impl(const std::string & module_name="root")
    {
        if (log <= level)
        {
            stream_ << log << ':' << module_name << ":";
        }
    }

    ~logger_impl()
    {
        if (log <= level)
        {
            log_type::log(stream_.str());
        }
    }

    std::ostream& get_stream()
    {
        return stream_;
    }

private:
    std::ostringstream stream_;
};

template<typename, typename log_level, log_level, unsigned>
class logger_null_stream_impl
{
    template<typename T>
    class null_buffer: public std::basic_streambuf<T>
    {
        std::streambuf::int_type overflow(std::streambuf::int_type)
        {
            return 0;
        }
    };

public:
    logger_null_stream_impl(const std::string&, unsigned, const std::string&)
        : null_log_(&null_buffer_)
    { }

    std::ostream& get_stream()
    {
        return null_log_;
    }

private:
    null_buffer<char> null_buffer_;
    std::ostream null_log_;
};

class clog
{
public:
    static void log(const std::string& log)
    {
        std::clog << log; // << std::endl;
    }
};

} // namespace logger


#ifndef LOG_LEVEL
# define LOG_LEVEL 3 //error, warn, debug
#endif

#ifndef LOG_TYPE
# define LOG_TYPE ::logger::clog
#endif

#define DEVLOG_INVOKER(logger_impl, log_type, log_level_type, given_level, show_level)\
    logger_impl<log_type, log_level_type, given_level, show_level>(__FILE__, __LINE__, __FUNCTION__).get_stream()

#ifdef NDEBUG
# define DEVLOG(given_level)\
    DEVLOG_INVOKER(::logger::logger_null_stream_impl, __logger_none_t, ::log_level, given_level, LOG_LEVEL)
#else
# define DEVLOG(given_level)\
    DEVLOG_INVOKER(::logger::logger_impl, LOG_TYPE, ::log_level, given_level, LOG_LEVEL)
#endif


#define LOG_INVOKER(logger_impl, log_type, log_level_type, given_level, module)\
    logger_impl<log_type, log_level_type, given_level, given_level>(module).get_stream()

#ifdef LOG_MODULE
#define LOG_(given_level)\
    LOG_INVOKER(::logger::logger_impl, LOG_TYPE, ::log_level, given_level, LOG_MODULE)
#else
#define LOG_(given_level)\
    LOG_INVOKER(::logger::logger_impl, LOG_TYPE, ::log_level, given_level, "root")
#endif

#define NULL_LOG() DEVLOG_INVOKER(::logger::logger_null_stream_impl, __logger_none_t, ::log_level, error, 100)


#define LOG(given_level) (verbose ? LOG_(given_level) : NULL_LOG())

#endif
