//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#version 330

// #define INT_MAX 2147483647
// #define INT_MIN -2147483648

uniform vec3 cam_pos;
uniform float MaxDepth;

in vec3 frag_position;

out uvec4 asdfFragColor;

uint float_to_normalized_fixed(float input) {
    // return 8346;
    // return 39294u;
    float normalized = input / MaxDepth / 2.0f;
    return uint(clamp(normalized+0.5f, 0.0f, 1.0f) * 4294967295.0f);
}


void main(void)
{
	// gl_asdfFragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
    // float mult = gl_FrontFacing ? -1 : 1;

    float dist = length(frag_position-cam_pos);
    // float dist = 100f;

    // if (!gl_FrontFacing) {
    //     asdfFragColor.rgba = uvec4(
    //         float_to_normalized_fixed(-dist),
    //         float_to_normalized_fixed(dist), 
    //         float_to_normalized_fixed(-MaxDepth), 
    //         float_to_normalized_fixed(-MaxDepth)
    //     );
    // } else {
    //     // gl_asdfFragColor.rgba = uvec4(-MaxDepth, -MaxDepth, -dist, dist);
    //     asdfFragColor.rgba = uvec4(
    //         float_to_normalized_fixed(-MaxDepth), 
    //         float_to_normalized_fixed(-MaxDepth),
    //         float_to_normalized_fixed(-dist),
    //         float_to_normalized_fixed(dist)
    //     );
    // }

    if (!gl_FrontFacing){
        dist = 100f;
        asdfFragColor.rgba = uvec4(
            float_to_normalized_fixed(-dist), 
            float_to_normalized_fixed(dist),
            float_to_normalized_fixed(-dist),
            float_to_normalized_fixed(dist)
        );
    } else {
        dist = 200f;
    asdfFragColor.rgba = uvec4(
        float_to_normalized_fixed(-dist), 
        float_to_normalized_fixed(dist),
        float_to_normalized_fixed(-dist),
        float_to_normalized_fixed(dist)
    );
    }

}

