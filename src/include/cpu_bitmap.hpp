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

#include "stb_image_write.h"
#include "utils.hpp"


struct CPUBitmap
{
    unsigned char *pixels;
    int            x, y;
    void          *dataBlock;
    void (*bitmapExit)(void *);
    std::string sourceFile;

    CPUBitmap(int width, int height, void *d = nullptr, const char *srcFile = nullptr)
    {
        pixels     = new unsigned char[width * height * 4];
        x          = width;
        y          = height;
        dataBlock  = d;
        bitmapExit = nullptr;
        sourceFile = srcFile ? std::string(srcFile) : std::string();
    }

    ~CPUBitmap() { delete[] pixels; }

    unsigned char *get_ptr(void) const { return pixels; }
    long           image_size(void) const { return x * y * 4; }

    void display_and_exit(void (*e)(void *) = nullptr)
    {
        CPUBitmap **bitmap = get_bitmap_ptr();
        *bitmap            = this;
        bitmapExit         = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int   c        = 1;
        char *dummy[1] = {nullptr};
        glutInit(&c, dummy);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
        glutInitWindowSize(x, y);
        glutCreateWindow("bitmap");
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

    bool save_to_png(const char *filename = nullptr) const
    {
        std::string output_path;
        if (filename) {
            output_path = filename;
        }
        else {
            output_path = generate_output_path(sourceFile.c_str());
        }

        char png_filename[256];
        snprintf(png_filename, sizeof(png_filename), "%s.png", output_path.c_str());

        return stbi_write_png(png_filename, x, y, 4, pixels, x * 4) != 0;
    }

    // static method used for glut callbacks
    static CPUBitmap **get_bitmap_ptr(void)
    {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y)
    {
        switch (key) {
        case 27:
            CPUBitmap *bitmap = *(get_bitmap_ptr());
            if (bitmap != nullptr && bitmap->dataBlock != nullptr && bitmap->bitmapExit != nullptr)
                bitmap->bitmapExit(bitmap->dataBlock);
            exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw(void)
    {
        CPUBitmap *bitmap = *(get_bitmap_ptr());
        if (bitmap != nullptr) {
            glClearColor(0.0, 0.0, 0.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
            glFlush();
        }
    }
};
