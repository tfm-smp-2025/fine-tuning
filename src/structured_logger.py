import os
import datetime
import logging
import time
import json
import traceback
import uuid

from typing import Optional, Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGGING_DIR = os.path.join(ROOT_DIR, "experiment-viewer", "logs")
PRINT_DEBUG = os.getenv("PRINT_DEBUG", "false") in ("1", "t", "true", "yes")
PRINT_INFO = os.getenv("PRINT_INFO", "true") in ("1", "t", "true", "yes")
PRINT_VERBOSE = os.getenv("PRINT_VERBOSE", "false") in ("1", "t", "true", "yes")


class StructuredLoggerContext:
    def __init__(
        self, parent: "StructuredLogger", context_name=None, context_params={}
    ):
        self.name = context_name
        self.parent = parent
        self.root_logger = logging.getLogger()
        self.context_params = dict(**context_params, id=str(uuid.uuid4()))
        self.previous_handler = None
        self.level = 0

    def __enter__(self):
        # Add itself as handler
        assert len(self.root_logger.handlers) in (0, 1)
        if len(self.root_logger.handlers) == 1:
            self.previous_handler = self.root_logger.handlers[0]
            self.root_logger.removeHandler(self.previous_handler)

        self.root_logger.addHandler(self)
        self.log_operation(
            "INFO",
            "Entering context: {}".format(self.name),
            operation="enter_context",
            data={
                "parent": self.parent.name,
                "parameters": self.context_params,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, tb):
        # Restore handler
        self.log_operation(
            "INFO",
            "Leaving context: {}".format(self.name),
            operation="leave_context",
            data={
                "parent": self.parent.name,
                "parameters": self.context_params,
            },
        )
        self.root_logger.removeHandler(self)
        if self.previous_handler is not None:
            self.root_logger.addHandler(self.previous_handler)

    def handle(self, record: logging.LogRecord):
        exception = None
        if record.exc_info is not None:
            exception = "\n".join(
                traceback.format_exception(
                    record.exc_info[0], record.exc_info[1], record.exc_info[2]
                )
            )
        elif record.levelname == "INFO" and record.getMessage().startswith(
            "HTTP Request: GET "
        ):
            # We can ignore these
            return

        self._log(
            record.levelname,
            record.getMessage(),
            operation="log",
            context=self,
            exception=exception,
        )

    def log_operation(
        self,
        level,
        message,
        operation,
        data,
        context=None,
        exception=None,
    ):
        self.parent.log_operation(
            level=level,
            message=message,
            operation=operation,
            data=data,
            context=context or self,
            exception=exception,
        )

    def _log(self, level, message, operation, context, exception=None):
        self.parent._log(level, message, operation, context, exception)


class StructuredLogger:
    def __init__(self):
        self.output_path = os.path.join(
            LOGGING_DIR, "log-" + str(datetime.datetime.now()) + ".jsonl"
        )
        self.name = "Root context"
        self.output = None
        self.info("Starting operation")

    def context(self, context_name=None, context_params={}):
        return StructuredLoggerContext(
            parent=self,
            context_name=context_name,
            context_params=context_params,
        )

    def debug(self, message, context=None):
        self._log("DEBUG", message, operation="log", context=context)

    def info(self, message, context=None):
        self._log("INFO", message, operation="log", context=context)

    def warn(self, message, context=None):
        self._log("WARN", message, operation="log", context=context)

    def error(self, message, context=None):
        self._log("ERROR", message, operation="log", context=context)

    def fatal(self, message, context=None):
        self._log("FATAL", message, operation="log", context=context)

    def log_operation(
        self,
        level,
        message,
        operation,
        data,
        context=None,
        exception=None,
    ):
        # TODO: Deduplicate with log?
        self._log(
            level=level,
            message=message,
            operation=operation,
            data=data,
            context=context,
            exception=exception,
        )

    def _log(self, level, message, operation, context, exception=None, data=None):
        log_data = data
        self._print_log(level, message, context, exception, data=data)

        data = {
            "level": level,
            "message": message,
            "operation": operation,
            "timestamp": time.time(),
            "time": datetime.datetime.now().isoformat(),
        }

        if exception is not None:
            data["exception"] = str(exception)

        if context is not None:
            data["context_name"] = context.name

        if log_data is not None:
            data["data"] = log_data

        with open(self.output_path, "at") as f:
            f.write(json.dumps(data) + "\n")

    def _print_log(self, level, message, context, exception=None, data=None):
        if level == "DEBUG" and not PRINT_DEBUG:
            return
        if level == "INFO" and not PRINT_INFO:
            return

        stime = datetime.datetime.now().isoformat()
        ctxt = ""
        if context:
            ctxt = f"[{context.name}]\t"
        print(f"{stime} {ctxt}{level}:\t \x1b[1m{message}\x1b[0m")

        if exception:
            print("---------- 8< ---------- EXCEPTION")
            print(exception)
            print("---------- >8 ----------")

        if data and PRINT_VERBOSE:
            # Minor formatting to avoid overwhelming output with data
            MAX_DATALINES = 40
            MAX_DATALINE_WIDTH = 100
            datalines_raw = json.dumps(data, indent=4).split("\n")
            datalines = []

            max_datalines_it1 = MAX_DATALINES
            if len(datalines_raw) > MAX_DATALINES:
                max_datalines_it1 = MAX_DATALINES - 2

            for dataline in datalines_raw[:MAX_DATALINES]:
                if len(dataline) > MAX_DATALINE_WIDTH:
                    dataline = dataline[: MAX_DATALINE_WIDTH - 3] + "..."
                datalines.append(dataline)

            if len(datalines_raw) > MAX_DATALINES:
                datalines.append("    ...")

                dataline = datalines_raw[-1]
                if len(dataline) > MAX_DATALINE_WIDTH:
                    dataline = dataline[: MAX_DATALINE_WIDTH - 3] + "..."
                datalines.append(dataline)

            print("---------- 8< ---------- DATA")
            print("\n".join(datalines))
            print("---------- >8 ----------")


LOGGER: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    global LOGGER
    if LOGGER is None:
        LOGGER = StructuredLogger()

    return LOGGER


def get_context() -> Union[StructuredLoggerContext, StructuredLogger]:
    root_logger = logging.getLogger()
    context_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, StructuredLoggerContext)
    ]

    if len(context_handlers) == 0:
        return get_logger()
    elif len(context_handlers) == 1:
        return context_handlers[0]
    else:
        logging.error(
            "Pulling context handlers, found >1 (defaulting to last): {}".format(
                context_handlers
            )
        )
        return context_handlers[-1]
