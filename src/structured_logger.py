import os
import datetime
import logging
import time
import json
import traceback

from typing import Optional

class StructuredLoggerContext:
    def __init__(self, parent: 'StructuredLogger', context_name=None):
        self.name = context_name
        self.parent = parent
        self.root_logger = logging.getLogger()
        self.previous_handler = None
        self.level = 0

    def __enter__(self):
        # Add itself as handler
        assert len(self.root_logger.handlers) in (0, 1)
        if len(self.root_logger.handlers) == 1:
            self.previous_handler = self.root_logger.handlers[0]
            self.root_logger.removeHandler(self.previous_handler)

        self.root_logger.addHandler(self)
        logging.info("Entering context")
        return self

    def __exit__(self, exc_type, exc_val, tb):
        # Restore handler
        logging.info("Leaving context")
        self.root_logger.removeHandler(self)
        if self.previous_handler is not None:
            self.root_logger.addHandler(self.previous_handler)

    def handle(self, record: logging.LogRecord):
        exception = None
        if record.exc_info is not None:
            exception = '\n'.join(
                traceback.format_exception(
                    record.exc_info[0], record.exc_info[1], record.exc_info[2]
                )
            )

        self._log(
            record.levelname,
            record.getMessage(),
            context=self,
            exception=exception,
        )

    def _log(self, level, message, context, exception=None):
        self.parent._log(level, message, context, exception)


class StructuredLogger:
    def __init__(self):
        self.output_path = os.path.abspath("log-" + str(datetime.datetime.now()) + ".jsonl")
        self.output = None
        self.info("Starting operation")

    def context(self, context_name=None):
        return StructuredLoggerContext(parent=self, context_name=context_name)

    def debug(self, message, context=None):
        self._log('debug', message, context)
    def info(self, message, context=None):
        self._log('info', message, context)
    def warn(self, message, context=None):
        self._log('warn', message, context)
    def error(self, message, context=None):
        self._log('error', message, context)
    def fatal(self, message, context=None):
        self._log('fatal', message, context)

    def _log(self, level, message, context, exception=None):
        self._print_log(level, message, context, exception)

        data = {
            "level": level,
            "message": message,
            "timestamp": time.time(),
            "time": datetime.datetime.now().isoformat(),
        }

        if exception is not None:
            data['exception'] = str(exception)

        if context is not None:
            data['context_name'] = context.name

        with open(self.output_path, 'at') as f:
            f.write(json.dumps(data) + '\n')

    def _print_log(self, level, message, context, exception=None):
        stime = datetime.datetime.now().isoformat()
        ctxt = ''
        if context:
            ctxt = f'[{context.name}]\t'
        print(f"{stime} {ctxt}{level}:\t \x1b[1m{message}\x1b[0m")
        if exception:
            print("---------- 8< ----------")
            print(exception)
            print("---------- >8 ----------")


LOGGER: Optional[StructuredLogger] = None

def get_logger() -> StructuredLogger:
    global LOGGER
    if LOGGER is None:
        LOGGER = StructuredLogger()

    return LOGGER
