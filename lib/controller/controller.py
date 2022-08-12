# -*- coding: utf-8 -*-
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#  Author: Mauro Soria

import os
import gc
import time
import re

from urllib.parse import urlparse

from lib.connection.dns import cache_dns
from lib.connection.requester import Requester
from lib.core.decorators import locked
from lib.core.dictionary import Dictionary, get_blacklists
from lib.core.exceptions import (
    InvalidRawRequest, InvalidURLException,
    RequestException, SkipTargetInterrupt,
    QuitInterrupt, UnpicklingError,
)
from lib.core.fuzzer import Fuzzer
from lib.core.logger import enable_logging, logger
from lib.core.settings import (
    BANNER, DEFAULT_HEADERS, DEFAULT_SESSION_FILE,
    EXTENSION_RECOGNITION_REGEX, MAX_CONSECUTIVE_REQUEST_ERRORS,
    NEW_LINE, SCRIPT_PATH, STANDARD_PORTS,
    PAUSING_WAIT_TIMEOUT, UNKNOWN
)
from lib.parse.rawrequest import parse_raw
from lib.parse.url import clean_path, parse_path
from lib.reports.csv_report import CSVReport
from lib.reports.html_report import HTMLReport
from lib.reports.json_report import JSONReport
from lib.reports.markdown_report import MarkdownReport
from lib.reports.plain_text_report import PlainTextReport
from lib.reports.simple_report import SimpleReport
from lib.reports.xml_report import XMLReport
from lib.reports.sqlite_report import SQLiteReport
from lib.utils.common import get_valid_filename, human_size, lstrip_once
from lib.utils.file import FileUtils
from lib.utils.pickle import pickle, unpickle
from lib.utils.schemedet import detect_scheme
from lib.analysis.analyzer import Analyzer


class Controller:
    def __init__(self, options, output):
        if options.session_file:
            self._import(options.session_file)
            self.old_session = True
        else:
            self.setup(options, output)
            self.old_session = False

        self.run()

    def _import(self, session_file):
        try:
            with open(session_file, "rb") as fd:
                indict, last_output = unpickle(fd)
        except UnpicklingError:
            self.output.error(
                f"{session_file} is not a valid session file or it's in an old format"
            )
            exit(1)

        self.__dict__ = {**indict, **vars(self)}

    def _export(self, session_file):
        self.current_job -= 1
        # Save written output
        last_output = self.output.buffer.rstrip()

        # Can't pickle Fuzzer class due to _thread.lock objects
        del self.fuzzer

        with open(session_file, "wb") as fd:
            pickle((vars(self), last_output), fd)

    def setup(self, options, output):
        self.options = options
        self.output = output

        if self.options.raw_file:
            try:
                self.options.update(
                    zip(
                        ["urls", "httpmethod", "headers", "data"],
                        parse_raw(self.options.raw_file),
                    )
                )
            except InvalidRawRequest as e:
                print(str(e))
                exit(1)
        else:
            self.options.headers = {**DEFAULT_HEADERS, **self.options.headers}

            if self.options.cookie:
                self.options.headers["Cookie"] = self.options.cookie
            if self.options.useragent:
                self.options.headers["User-Agent"] = self.options.useragent

        self.random_agents = None
        if self.options.use_random_agents:
            self.random_agents = FileUtils.get_lines(
                FileUtils.build_path(SCRIPT_PATH, "db", "user-agents.txt")
            )

        self.requester = Requester(
            max_pool=self.options.threads_count,
            max_retries=self.options.max_retries,
            max_rate=self.options.max_rate,
            timeout=self.options.timeout,
            proxy=self.options.proxy,
            follow_redirects=self.options.follow_redirects,
            httpmethod=self.options.httpmethod,
            headers=self.options.headers,
            data=self.options.data,
            random_agents=self.random_agents,
            cert_file=self.options.cert_file,
            key_file=self.options.key_file,
        )
        self.dictionary = Dictionary(
            files=self.options.wordlists,
            extensions=self.options.extensions,
            suffixes=self.options.suffixes,
            prefixes=self.options.prefixes,
            lowercase=self.options.lowercase,
            uppercase=self.options.uppercase,
            capitalization=self.options.capitalization,
            force_extensions=self.options.force_extensions,
            overwrite_extensions=self.options.overwrite_extensions,
            exclude_extensions=self.options.exclude_extensions,
            remove_extensions=self.options.remove_extensions,
        )
        self.blacklists = get_blacklists(self.options.extensions)
        self.results = []
        self.responses = []
        self.targets = options.urls
        self.start_time = time.time()
        self.passed_urls = set()
        self.directories = []
        self.report = None
        self.batch = False
        self.current_job = 0
        self.errors = 0
        self.consecutive_errors = 0

        if self.options.auth:
            self.requester.set_auth(self.options.auth_type, self.options.auth)

        if self.options.proxy_auth:
            self.requester.set_proxy_auth(self.options.proxy_auth)

        if self.options.log_file:
            self.options.log_file = FileUtils.get_abs_path(self.options.log_file)

            try:
                FileUtils.create_dir(FileUtils.parent(self.options.log_file))
                if not FileUtils.can_write(self.options.log_file):
                    raise Exception

                enable_logging(self.options.log_file, self.options.log_file_size or 0)

            except Exception:
                self.output.error(
                    f"Couldn't create log file at {self.options.log_file}"
                )
                exit(1)

        if self.options.autosave_report:
            self.report_path = self.options.output_path or FileUtils.build_path(
                SCRIPT_PATH, "reports"
            )

            try:
                FileUtils.create_dir(self.report_path)
                if not FileUtils.can_write(self.report_path):
                    raise Exception

            except Exception:
                self.output.error(
                    f"Couldn't create report folder at {self.report_path}"
                )
                exit(1)

        self.output.header(BANNER)
        self.output.config(
            ", ".join(self.options["extensions"]),
            ", ".join(self.options["prefixes"]),
            ", ".join(self.options["suffixes"]),
            str(self.options["threads_count"]),
            str(len(self.dictionary)),
            str(self.options["httpmethod"]),
        )

        self.setup_reports()

        if self.options.log_file:
            self.output.log_file(self.options.log_file)

    def run(self):
        # match_callbacks and not_found_callbacks callback values:
        #  - *args[0]: lib.connection.Response() object
        #
        # error_callbacks callback values:
        #  - *args[0]: exception
        match_callbacks = (
            self.match_callback, self.reset_consecutive_errors
        )
        not_found_callbacks = (
            self.update_progress_bar, self.reset_consecutive_errors
        )
        error_callbacks = (self.raise_error, self.append_error_log)

        while self.targets:
            url = self.targets[0]
            self.fuzzer = Fuzzer(
                self.requester,
                self.dictionary,
                suffixes=self.options.suffixes,
                prefixes=self.options.prefixes,
                exclude_response=self.options.exclude_response,
                threads=self.options.threads_count,
                delay=self.options.delay,
                crawl=self.options.crawl,
                match_callbacks=match_callbacks,
                not_found_callbacks=not_found_callbacks,
                error_callbacks=error_callbacks,
            )

            try:
                self.set_target(url)

                if not self.directories:
                    for subdir in self.options.subdirs:
                        self.add_directory(self.base_path + subdir)

                if not self.old_session:
                    self.output.target(self.url)

                self.start()

            except (
                InvalidURLException,
                RequestException,
                SkipTargetInterrupt,
                KeyboardInterrupt,
            ) as e:
                self.directories.clear()
                self.dictionary.reset()

                if e.args:
                    self.output.error(str(e))

            except QuitInterrupt as e:
                self.output.error(e.args[0])
                exit(0)

            finally:
                self.targets.pop(0)

        self.output.warning("\nScan Task Completed, Starting Deep Analysis ...")

        analyzer = Analyzer(self.options, self.output, self.report)
        analyzer.analysis_responses(self.responses)

        self.output.warning("\nTask Completed")

        if self.options.session_file:
            try:
                os.remove(self.options.session_file)
            except Exception:
                self.output.error("Failed to delete old session file, remove it to free some space")

    def start(self):
        while self.directories:
            try:
                gc.collect()

                self.current_job += 1
                current_directory = self.directories[0]

                if not self.old_session:
                    current_time = time.strftime("%H:%M:%S")
                    msg = f"{NEW_LINE}[{current_time}] Starting: {current_directory}"

                    self.output.warning(msg)

                self.fuzzer.set_base_path(current_directory)
                self.fuzzer.start()
                self.process()

            except KeyboardInterrupt:
                pass

            finally:
                self.dictionary.reset()
                self.directories.pop(0)

                self.old_session = False

    def set_target(self, url):
        # If no scheme specified, unset it first
        if "://" not in url:
            url = f"{self.options.scheme or UNKNOWN}://{url}"
        if not url.endswith("/"):
            url += "/"

        parsed = urlparse(url)
        self.base_path = lstrip_once(parsed.path, "/")

        # Credentials in URL
        if "@" in parsed.netloc:
            cred, parsed.netloc = parsed.netloc.split("@")
            self.requester.set_auth("basic", cred)

        host = parsed.netloc.split(":")[0]

        if parsed.scheme not in (UNKNOWN, "https", "http"):
            raise InvalidURLException(f"Unsupported URI scheme: {parsed.scheme}")

        # If no port specified, set default (80, 443)
        try:
            port = int(parsed.netloc.split(":")[1])

            if not 0 < port < 65536:
                raise ValueError
        except IndexError:
            port = STANDARD_PORTS.get(parsed.scheme, None)
        except ValueError:
            port = parsed.netloc.split(":")[1]
            raise InvalidURLException(f"Invalid port number: {port}")

        if self.options.ip:
            cache_dns(host, port, self.options.ip)

        try:
            # If no scheme is found, detect it by port number
            scheme = (
                parsed.scheme
                if parsed.scheme != UNKNOWN
                else detect_scheme(host, port)
            )
        except ValueError:
            # If the user neither provides the port nor scheme, guess them based
            # on standard website characteristics
            scheme = detect_scheme(host, 443)
            port = STANDARD_PORTS[scheme]

        self.url = f"{scheme}://{host}"

        if port != STANDARD_PORTS[scheme]:
            self.url += f":{port}"

        self.url += "/"

        self.requester.set_url(self.url)

    def setup_batch_reports(self):
        """Create batch report folder"""

        self.batch = True
        current_time = time.strftime("%y-%m-%d_%H-%M-%S")
        batch_session = f"BATCH-{current_time}"
        batch_directory_path = FileUtils.build_path(self.report_path, batch_session)

        try:
            FileUtils.create_dir(batch_directory_path)
        except Exception:
            self.output.error(f"Couldn't create batch folder at {batch_directory_path}")
            exit(1)

        return batch_directory_path

    def get_output_extension(self):
        if self.options.output_format in ("plain", "simple"):
            return ".txt"

        return f".{self.options.output_format}"

    def setup_reports(self):
        """Create report file"""

        output_file = self.options.output_file

        if self.options.autosave_report and not output_file:
            if len(self.targets) > 1:
                directory_path = self.setup_batch_reports()
                filename = "BATCH" + self.get_output_extension()
            else:
                parsed = urlparse(self.targets[0])

                if not parsed.netloc:
                    parsed = urlparse(f"//{self.targets[0]}")

                filename = get_valid_filename(f"{parsed.path}_")
                filename += time.strftime("%y-%m-%d_%H-%M-%S")
                filename += self.get_output_extension()
                directory_path = FileUtils.build_path(
                    self.report_path, get_valid_filename(f"{parsed.scheme}_{parsed.netloc}")
                )

            output_file = FileUtils.get_abs_path((FileUtils.build_path(directory_path, filename)))

            if FileUtils.exists(output_file):
                i = 2
                while FileUtils.exists(f"{output_file}_{i}"):
                    i += 1

                output_file += f"_{i}"

            try:
                FileUtils.create_dir(directory_path)
            except Exception:
                self.output.error(
                    f"Couldn't create the reports folder at {directory_path}"
                )
                exit(1)

        if not output_file:
            return

        if self.options.output_format == "plain":
            self.report = PlainTextReport(output_file)
        elif self.options.output_format == "json":
            self.report = JSONReport(output_file)
        elif self.options.output_format == "xml":
            self.report = XMLReport(output_file)
        elif self.options.output_format == "md":
            self.report = MarkdownReport(output_file)
        elif self.options.output_format == "csv":
            self.report = CSVReport(output_file)
        elif self.options.output_format == "html":
            self.report = HTMLReport(output_file)
        elif self.options.output_format == "sqlite":
            self.report = SQLiteReport(output_file)
        else:
            self.report = SimpleReport(output_file)

        self.output.output_file(output_file)

    def is_valid(self, res):
        """Validate the response by different filters"""

        if res.status in self.options.exclude_status_codes:
            return False

        if res.status not in (self.options.include_status_codes or range(100, 1000)):
            return False

        if (
            res.status in self.blacklists
            and any(
                res.path.endswith(lstrip_once(suffix, "/"))
                for suffix in self.blacklists.get(res.status)
            )
        ):
            return False

        if human_size(res.length).rstrip() in self.options.exclude_sizes:
            return False

        if res.length < self.options.minimum_response_size:
            return False

        if res.length > self.options.maximum_response_size > 0:
            return False

        if any(ex_text in res.content for ex_text in self.options.exclude_texts):
            return False

        if self.options.exclude_regex and re.search(
            self.options.exclude_regex, res.content
        ):
            return False

        if self.options.exclude_redirect and (
            self.options.exclude_redirect in res.redirect
            or re.search(self.options.exclude_redirect, res.redirect)
        ):
            return False

        return True

    def reset_consecutive_errors(self, response):
        self.responses.append(response)  # 记录所有没有抛出异常的response
        self.consecutive_errors = 0

    def match_callback(self, response):
        if response.status in self.options.skip_on_status:
            raise SkipTargetInterrupt(
                f"Skipped the target due to {response.status} status code"
            )

        if not self.is_valid(response):
            return

        self.output.status_report(response, self.options.full_url)

        if response.status in self.options.recursion_status_codes and any(
            (
                self.options.recursive,
                self.options.deep_recursive,
                self.options.force_recursive,
            )
        ):
            if response.redirect:
                new_path = clean_path(parse_path(response.redirect))
                added_to_queue = self.recur_for_redirect(response.path, new_path)
            elif len(response.history):
                old_path = clean_path(parse_path(response.history[0]))
                added_to_queue = self.recur_for_redirect(old_path, response.path)
            else:
                added_to_queue = self.recur(response.path)

            if added_to_queue:
                self.output.new_directories(added_to_queue)

        if self.options.replay_proxy:
            # Replay the request with new proxy
            self.requester.request(response.full_path, proxy=self.options.replay_proxy)

        if self.report:
            self.results.append(response)
            self.report.save(self.results)

    def update_progress_bar(self, response):
        jobs_count = (
            len(self.options.subdirs) * (len(self.targets) - 1)
            + len(self.directories)
        )

        self.output.last_path(
            self.dictionary.index,
            len(self.dictionary),
            self.current_job,
            jobs_count,
            self.requester.rate,
            self.errors,
        )

    def raise_error(self, exception):
        if self.options.exit_on_error:
            raise QuitInterrupt("Canceled due to an error")

        self.errors += 1
        self.consecutive_errors += 1

        if self.consecutive_errors > MAX_CONSECUTIVE_REQUEST_ERRORS:
            raise SkipTargetInterrupt("Too many request errors")

    def append_error_log(self, exception):
        logger.exception(exception)

    def handle_pause(self):
        self.output.warning(
            "CTRL+C detected: Pausing threads, please wait...", do_save=False
        )
        self.fuzzer.pause()

        start_time = time.time()
        while True:
            is_timed_out = time.time() - start_time > PAUSING_WAIT_TIMEOUT
            if self.fuzzer.is_stopped() or is_timed_out:
                break

            time.sleep(0.2)

        while True:
            msg = "[q]uit / [c]ontinue"

            if len(self.directories) > 1:
                msg += " / [n]ext"

            if len(self.targets) > 1:
                msg += " / [s]kip target"

            self.output.in_line(msg + ": ")

            option = input()

            if option.lower() == "q":
                self.output.in_line("[s]ave / [q]uit without saving: ")

                option = input()

                if option.lower() == "s":
                    msg = f"Save to file [{self.options.session_file or DEFAULT_SESSION_FILE}]: "

                    self.output.in_line(msg)

                    session_file = (
                        input() or self.options.session_file or DEFAULT_SESSION_FILE
                    )

                    self._export(session_file)
                    raise QuitInterrupt(f"Session saved to: {session_file}")
                elif option.lower() == "q":
                    raise QuitInterrupt("Canceled by the user")

            elif option.lower() == "c":
                self.fuzzer.resume()
                return

            elif option.lower() == "n" and len(self.directories) > 1:
                self.fuzzer.stop()
                return

            elif option.lower() == "s" and len(self.targets) > 1:
                raise SkipTargetInterrupt("Target skipped by the user")

    def is_timed_out(self):
        return time.time() - self.start_time > self.options.maxtime > 0

    def process(self):
        while True:
            try:
                while not self.fuzzer.wait(0.25):
                    if self.is_timed_out():
                        raise SkipTargetInterrupt(
                            "Runtime exceeded the maximum set by the user"
                        )

                break

            except KeyboardInterrupt:
                self.handle_pause()

    def add_directory(self, path):
        """Add directory to the recursion queue"""

        # Pass if path is in exclusive directories
        if any(
            "/" + dir in path for dir in self.options.exclude_subdirs
        ):
            return

        url = self.url + path

        if (
            path.count("/") - self.base_path.count("/") > self.options.recursion_depth > 0
            or url in self.passed_urls
        ):
            return

        self.directories.append(path)
        self.passed_urls.add(url)

    @locked
    def recur(self, path):
        dirs_count = len(self.directories)
        path = clean_path(path)

        if self.options.force_recursive and not path.endswith("/"):
            path += "/"

        if self.options.deep_recursive:
            i = 0
            for _ in range(path.count("/")):
                i = path.index("/", i) + 1
                self.add_directory(path[:i])
        elif (
            self.options.recursive
            and path.endswith("/")
            and re.search(EXTENSION_RECOGNITION_REGEX, path[:-1]) is None
        ):
            self.add_directory(path)

        # Return newly added directories
        return self.directories[dirs_count:]

    def recur_for_redirect(self, path, redirect_path):
        if redirect_path == path + "/":
            return self.recur(redirect_path)

        return []
