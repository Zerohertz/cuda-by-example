/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>
#include <cstdio>
#include <vector>

#include "gif.h"
#include "stb_image_write.h"
#include "utils.hpp"


struct CPUAnimBitmap
{
    unsigned char *pixels;
    int            width, height;
    void          *dataBlock;
    void (*fAnim)(void *, int);
    void (*animExit)(void *);
    void (*clickDrag)(void *, int, int, int, int);
    int dragStartX, dragStartY;

    std::vector<std::vector<unsigned char>> frames;
    bool                                    recording;
    int                                     frameDelay;
    std::string                             sourceFile;

    CPUAnimBitmap(int w, int h, void *d = nullptr, const char *srcFile = nullptr)
    {
        width      = w;
        height     = h;
        pixels     = new unsigned char[width * height * 4];
        dataBlock  = d;
        clickDrag  = nullptr;
        recording  = false;
        frameDelay = 100;
        sourceFile = srcFile ? std::string(srcFile) : std::string();
    }

    ~CPUAnimBitmap() { delete[] pixels; }

    unsigned char *get_ptr(void) const { return pixels; }
    long           image_size(void) const { return width * height * 4; }

    void click_drag(void (*f)(void *, int, int, int, int)) { clickDrag = f; }

    void start_recording(int delay = 100)
    {
        recording  = true;
        frameDelay = delay;
        frames.clear();
    }
    void stop_recording() { recording = false; }

    size_t frame_count() const { return frames.size(); }

    void capture_frame()
    {
        if (recording) {
            std::vector<unsigned char> frame(pixels, pixels + width * height * 4);
            frames.push_back(frame);
        }
    }

    bool save_to_gif(const char *filename = nullptr) const
    {
        if (frames.empty())
            return false;

        std::string output_path;
        if (filename) {
            output_path = filename;
        }
        else {
            output_path = generate_output_path(sourceFile.c_str());
        }

        char gif_filename[256];
        snprintf(gif_filename, sizeof(gif_filename), "%s.gif", output_path.c_str());

        GifWriter g;
        if (!GifBegin(&g, gif_filename, width, height, frameDelay / 10)) {
            return false;
        }
        for (size_t frame_idx = 0; frame_idx < frames.size(); ++frame_idx) {
            const auto &frame = frames[frame_idx];
            if (!GifWriteFrame(&g, frame.data(), width, height, frameDelay / 10)) {
                GifEnd(&g);
                return false;
            }
        }

        return GifEnd(&g);
    }

    void anim_and_exit(void (*f)(void *, int), void (*e)(void *))
    {
        CPUAnimBitmap **bitmap = get_bitmap_ptr();
        *bitmap                = this;
        fAnim                  = f;
        animExit               = e;

        // Start recording for GIF generation
        start_recording(5); // 50ms per frame

        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int   c        = 1;
        char *dummy[1] = {nullptr};
        glutInit(&c, dummy);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(width, height);
        glutCreateWindow("bitmap");
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        if (clickDrag != nullptr)
            glutMouseFunc(mouse_func);
        glutIdleFunc(idle_func);
        glutMainLoop();
    }

    // static method used for glut callbacks
    static CPUAnimBitmap **get_bitmap_ptr(void)
    {
        static CPUAnimBitmap *gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func(int button, int state, int mx, int my)
    {
        if (button == GLUT_LEFT_BUTTON) {
            CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            }
            else if (state == GLUT_UP) {
                bitmap->clickDrag(bitmap->dataBlock, bitmap->dragStartX, bitmap->dragStartY, mx, my);
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func(void)
    {
        static int     ticks  = 1;
        CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
        bitmap->fAnim(bitmap->dataBlock, ticks++);
        bitmap->capture_frame(); // Automatically capture frames during animation
        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y)
    {
        switch (key) {
        case 27: // ESC key
            CPUAnimBitmap *bitmap = *(get_bitmap_ptr());

            // Stop recording and save GIF
            bitmap->stop_recording();
            if (bitmap->frame_count() > 0) {
                printf("Saving GIF with %zu frames...\n", bitmap->frame_count());
                bitmap->save_to_gif();
                printf("GIF saved successfully!\n");
            }

            bitmap->animExit(bitmap->dataBlock);
            // delete bitmap;
            exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw(void)
    {
        CPUAnimBitmap *bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
        glutSwapBuffers();
    }
};
