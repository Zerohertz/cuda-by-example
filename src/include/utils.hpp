#pragma once

#include <string>
#include <unistd.h>

inline std::string generate_output_path(const char *source_file)
{
    std::string file_path(source_file);
    if (file_path[0] != '/') {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            file_path = std::string(cwd) + "/" + file_path;
        }
    }
    size_t src_pos = file_path.find("/src/");
    if (src_pos == std::string::npos) {
        return "./res";
    }
    std::string project_home = file_path.substr(0, src_pos);
    size_t      last_slash   = file_path.find_last_of('/');
    std::string filename     = file_path.substr(last_slash + 1);
    size_t      dot_pos      = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        filename = filename.substr(0, dot_pos);
    }
    std::string chapter;
    size_t      chapter_pos = file_path.find("/chapter");
    if (chapter_pos != std::string::npos) {
        size_t chapter_start = chapter_pos + 1;
        size_t chapter_end   = file_path.find('/', chapter_start);
        if (chapter_end != std::string::npos) {
            chapter = file_path.substr(chapter_start, chapter_end - chapter_start);
        }
    }
    if (chapter.empty()) {
        chapter = "unknown";
    }
    return project_home + "/res/" + chapter + "_" + filename;
}
