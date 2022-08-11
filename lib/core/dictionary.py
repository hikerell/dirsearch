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

import re

from lib.core.decorators import locked
from lib.core.settings import (
    SCRIPT_PATH, EXTENSION_TAG, EXCLUDE_OVERWRITE_EXTENSIONS,
    EXTENSION_RECOGNITION_REGEX, EXTENSION_REGEX
)
from lib.parse.url import clean_path
from lib.utils.common import lstrip_once
from lib.utils.file import FileUtils


# Get ignore paths for status codes.
# Reference: https://github.com/maurosoria/dirsearch#Blacklist
def get_blacklists(extensions):
    blacklists = {}

    for status in [400, 403, 500]:
        blacklist_file_name = FileUtils.build_path(SCRIPT_PATH, "db")
        blacklist_file_name = FileUtils.build_path(
            blacklist_file_name, f"{status}_blacklist.txt"
        )

        if not FileUtils.can_read(blacklist_file_name):
            # Skip if cannot read file
            continue

        blacklists[status] = Dictionary(
            files=[blacklist_file_name],
            extensions=extensions
        )

    return blacklists


class Dictionary:
    def __init__(self, **kwargs):
        self._entries = []
        self._index = 0
        self._dictionary_files = kwargs.get("files", set())
        self.extensions = kwargs.get("extensions", ())
        self.exclude_extensions = kwargs.get("exclude_extensions", ())
        self.prefixes = kwargs.get("prefixes", ())
        self.suffixes = kwargs.get("suffixes", ())
        self.force_extensions = kwargs.get("force_extensions", False)
        self.overwrite_extensions = kwargs.get("overwrite_extensions", False)
        self.remove_extensions = kwargs.get("remove_extensions", False)
        self.lowercase = kwargs.get("lowercase", False)
        self.uppercase = kwargs.get("uppercase", False)
        self.capitalization = kwargs.get("capitalization", False)
        self.generate()

    @property
    def index(self):
        return self._index

    def generate(self):
        """
        Dictionary.generate() behaviour

        Classic dirsearch wordlist:
          1. If %EXT% keyword is present, append one with each extension REPLACED.
          2. If the special word is no present, append line unmodified.

        Forced extensions wordlist (NEW):
          This type of wordlist processing is a mix between classic processing
          and DirBuster processing.
              1. If %EXT% keyword is present in the line, immediately process as "classic dirsearch" (1).
              2. If the line does not include the special word AND is NOT terminated by a slash,
                append one with each extension APPENDED (line.ext) and ONLYE ONE with a slash.
              3. If the line does not include the special word and IS ALREADY terminated by slash,
                append line unmodified.
        """

        re_ext_tag = re.compile(EXTENSION_TAG, re.IGNORECASE)
        re_extension = re.compile(EXTENSION_REGEX, re.IGNORECASE)

        for dict_file in self._dictionary_files:
            for line in FileUtils.get_lines(dict_file):
                # Removing leading "/" to work with prefixes later
                line = lstrip_once(line, "/")

                if self.remove_extensions:
                    line = line.split(".")[0]

                if not self.is_valid(line):
                    continue

                # Classic dirsearch wordlist processing (with %EXT% keyword)
                if EXTENSION_TAG in line.lower():
                    for extension in self.extensions:
                        newline = re_ext_tag.sub(extension, line)
                        self.add(newline)
                # If "forced extensions" is used and the path is not a directory (terminated by /)
                # or has had an extension already, append extensions to the path
                elif (
                    self.force_extensions
                    and not line.endswith("/")
                    and not re.search(EXTENSION_RECOGNITION_REGEX, line)
                ):
                    self.add(line)
                    self.add(line + "/")

                    for extension in self.extensions:
                        self.add(f"{line}.{extension}")
                # Overwrite unknown extensions with selected ones (but also keep the origin)
                elif (
                    self.overwrite_extensions
                    and not line.endswith(self.extensions + EXCLUDE_OVERWRITE_EXTENSIONS)
                    # Paths that have queries in wordlist are usually used for exploiting
                    # diclosed vulnerabilities of services, skip such paths
                    and "?" not in line
                    and "#" not in line
                    and re.search(EXTENSION_RECOGNITION_REGEX, line)
                ):
                    self.add(line)

                    for extension in self.extensions:
                        newline = re_extension.sub(f".{extension}", line)
                        self.add(newline)
                # Append line unmodified.
                else:
                    self.add(line)

    def is_valid(self, path):
        # Skip comments and empty lines
        if not path or path.startswith("#"):
            return False

        # Skip if the path has excluded extensions
        cleaned_path = clean_path(path)
        if cleaned_path.endswith(
            tuple(f".{extension}" for extension in self.exclude_extensions)
        ):
            return False

        return True

    def add(self, path):
        def append(path):
            if self.lowercase:
                path = path.lower()
            elif self.uppercase:
                path = path.upper()
            elif self.capitalization:
                path = path.capitalize()

            if path not in self._entries:
                self._entries.append(path)

        for pref in self.prefixes:
            if not path.startswith(("/", pref)):
                append(pref + path)
        for suff in self.suffixes:
            if not path.endswith(("/", suff)) and "#" not in path:
                append(path + suff)

        if not self.prefixes and not self.suffixes:
            append(path)

    def reset(self):
        self._index = 0

    def __contains__(self, item):
        return item in self._entries

    def __getstate__(self):
        return (self._entries, self._index, self.extensions)

    def __setstate__(self, state):
        self._entries, self._index, self.extensions = state

    @locked
    def __next__(self):
        try:
            path = self._entries[self._index]
        except IndexError:
            raise StopIteration

        self._index += 1

        return path

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)
