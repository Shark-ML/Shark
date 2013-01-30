Shark Logging Facilities
========================

The Shark machine learning library offers a framework for logging
purposes as a part of its default installation (see
:doxy:`Logger.h`). Its purpose is twofold: On the one hand, debug,
status and error information can be shown and stored persistently in
an abstract manner. On the other hand, external components like GUI
framework can rely on the framework to redirect Shark logging messages
to custom handlers, e.g., GUI log views.

The design of the logging framework is simple and straightforward. An
instance of class :doxy:`Logger` is required to report messages to the
user. Every logger instance relies on a set of handlers (see
:doxy:`Logger::AbstractHandler`) for formatting and reporting
purposes. Loggers are announced within the whole library and queryable
by their name by means of the singleton :doxy:`LoggerPool`.
