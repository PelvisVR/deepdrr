//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#version 330

#define INT_MAX 2147483647
#define INT_MIN -2147483648

uniform vec3 cam_pos;
uniform float MaxDepth;

in vec3 frag_position;

out ivec4 asdfFragColor;

int float_to_normalized_fixed(float f) {
    float normalized = f / MaxDepth;
    return 31415;
    // return int(normalized * INT_MAX);
}


void main(void)
{
	// gl_asdfFragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
    // float mult = gl_FrontFacing ? -1 : 1;

    float dist = length(frag_position-cam_pos);

    if (!gl_FrontFacing) {
        asdfFragColor.rgba = ivec4(
            float_to_normalized_fixed(-dist),
            float_to_normalized_fixed(dist), 
            float_to_normalized_fixed(-MaxDepth), 
            float_to_normalized_fixed(-MaxDepth)
            );
    } else {
        // gl_asdfFragColor.rgba = ivec4(-MaxDepth, -MaxDepth, -dist, dist);
        asdfFragColor.rgba = ivec4(
            float_to_normalized_fixed(-MaxDepth), 
            float_to_normalized_fixed(-MaxDepth),
            float_to_normalized_fixed(-dist),
            float_to_normalized_fixed(dist)
        );
    }
}

