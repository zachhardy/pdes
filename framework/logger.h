#pragma once
#include <iostream>
#include <sstream>
#include <string>


namespace pdes
{
  class Logger;

  enum class LogLevel
  {
    SILENT,
    ERROR,
    WARNING,
    SUMMARY,
    ITERATION,
    DEBUG
  };

  class LoggerStream
  {
  public:
    LoggerStream(const Logger& logger, LogLevel level);
    ~LoggerStream();

    template<typename T>
    LoggerStream& operator<<(const T& value);

  private:
    const Logger& logger_;
    LogLevel level_;
    std::ostringstream buffer_;
  };

  class Logger
  {
  public:
    explicit Logger(std::ostream& out = std::cout, LogLevel level = LogLevel::SILENT);

    void set_level(const LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    LoggerStream log(const LogLevel level) const { return LoggerStream(*this, level); }

    LoggerStream error() const { return log(LogLevel::ERROR); }
    LoggerStream warning() const { return log(LogLevel::WARNING); }
    LoggerStream summary() const { return log(LogLevel::SUMMARY); }
    LoggerStream iter() const { return log(LogLevel::ITERATION); }
    LoggerStream debug() const { return log(LogLevel::DEBUG); }

    bool should_log(const LogLevel level) const { return level <= level_; }

    void write(LogLevel level, const std::string& message) const;

  private:
    static std::string to_label(LogLevel level);

    std::ostream& out_;
    LogLevel level_;

  public:
    static Logger& default_logger();
  };

  template<typename T>
  LoggerStream&
  LoggerStream::operator<<(const T& value)
  {
    buffer_ << value;
    return *this;
  }

  inline
  LoggerStream::LoggerStream(const Logger& logger, const LogLevel level)
    : logger_(logger),
      level_(level) {}

  inline
  LoggerStream::~LoggerStream()
  {
    logger_.write(level_, buffer_.str());
  }

  inline
  Logger::Logger(std::ostream& out, const LogLevel level)
    : out_(out),
      level_(level)
  {
  }

  inline void
  Logger::write(const LogLevel level, const std::string& message) const
  {
    {
      if (should_log(level))
        out_ << "[" << to_label(level) << "] " << message << std::endl;
    }
  }

  inline Logger&
  Logger::default_logger()
  {
    static Logger instance;
    return instance;
  }

  inline std::string
  Logger::to_label(const LogLevel level)
  {
    switch (level)
    {
      case LogLevel::WARNING: return "WARNING";
      case LogLevel::ITERATION: return "ITER";
      case LogLevel::SUMMARY: return "SUMMARY";
      case LogLevel::DEBUG: return "DEBUG";
      default: return "UNKNOWN";
    }
  }
}
