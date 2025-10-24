# This file is adapted from https://github.com/jennyzzt/dgm.

import tempfile
from pathlib import Path

import pytest

from tools.edit import tool_function


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file with content for testing."""
    file_path = temp_dir / "test.txt"
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    file_path.write_text(content)
    return file_path


class TestEditorTool:
    def test_view_file(self, sample_file):
        """Test viewing entire file content."""
        result = tool_function("view", str(sample_file))
        assert "line 1" in result
        assert "line 5" in result
        assert "Here's the result of running `cat -n`" in result

    def test_create_file(self, temp_dir):
        """Test creating a new file."""
        new_file = temp_dir / "new.txt"
        content = "test content\nline 2"
        result = tool_function("create", str(new_file), file_text=content)
        assert "File created successfully" in result
        assert new_file.read_text() == content

    def test_create_existing_file(self, sample_file):
        """Test attempting to create an already existing file."""
        result = tool_function("create", str(sample_file), file_text="new content")
        assert "Error" in result
        assert "already exists" in result

    def test_edit_file(self, sample_file):
        """Test editing an existing file."""
        new_content = "edited content\nnew line"
        result = tool_function("edit", str(sample_file), file_text=new_content)
        assert "has been overwritten" in result
        assert sample_file.read_text() == new_content

    def test_edit_nonexistent_file(self, temp_dir):
        """Test attempting to edit a nonexistent file."""
        non_existent_file = temp_dir / "does_not_exist.txt"
        result = tool_function("edit", str(non_existent_file), file_text="new content")
        assert "Error" in result
        assert "does not exist" in result

    def test_view_directory(self, temp_dir):
        """Test viewing directory contents."""
        # Create some files in the directory
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").touch()

        result = tool_function("view", str(temp_dir))
        assert "files and directories" in result
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    def test_invalid_path(self):
        """Test operations with invalid path."""
        result = tool_function("view", "/nonexistent/path")
        assert "Error" in result
        assert "does not exist" in result

    @pytest.mark.parametrize("command", ["unknown_command", "", None])
    def test_invalid_commands(self, command, sample_file):
        """Test various invalid commands."""
        result = tool_function(command, str(sample_file))
        assert "Error" in result
