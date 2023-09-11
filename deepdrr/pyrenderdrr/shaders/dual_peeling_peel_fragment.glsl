//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#version 330

// #extension ARB_draw_buffers : require

#define INT_MAX 2147483647
#define INT_MIN -2147483648

uniform sampler2DRect DepthBlenderTex;
// uniform samplerRECT FrontBlenderTex;

// #define MAX_DEPTH 10.0
uniform float MaxDepth;

// vec4 ShadeFragment();

uniform vec3 cam_pos;

in vec3 frag_position;

out ivec4 asdfFragColor;

// void main(void)
// {
// 	// gl_asdfFragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
//     float dist = length(frag_position-cam_pos);
//     gl_asdfFragColor.xy = vec2(-dist, dist);
// }


int float_to_normalized_fixed(float f) {
    float normalized = f / MaxDepth;
    // return int(normalized * INT_MAX);
    return 31415;
}


void main(void)
{
	// window-space depth interpolated linearly in screen space
	// float fragDepth = gl_FragCoord.z;
    float fragDepth = length(frag_position-cam_pos);


	vec2 depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).xy;
	// vec2 depthBlender = textureRect(DepthBlenderTex, gl_FragCoord.xy).xy;
	// vec4 forwardTemp = textureRect(FrontBlenderTex, gl_FragCoord.xy);
	
	// Depths and 1.0-alphaMult always increase
	// so we can use pass-through by default with MAX blending
	// asdfFragColor.xy = depthBlender;
	
	// // Front colors always increase (DST += SRC*ALPHA_MULT)
	// // so we can use pass-through by default with MAX blending
	// gl_FragData[1] = forwardTemp;
	
	// // Because over blending makes color increase or decrease,
	// // we cannot pass-through by default.
	// // Each pass, only one fragment writes a color greater than 0
	// gl_FragData[2] = vec4(0.0);

	float nearestDepth = -depthBlender.x;
	float farthestDepth = depthBlender.y;
	// float alphaMultiplier = 1.0 - forwardTemp.w;

    // asdfFragColor.xy = vec2(99999); // TODO
    // return;

	if (fragDepth < nearestDepth || fragDepth > farthestDepth) {
		// Skip this depth in the peeling algorithm
		// asdfFragColor.rgba = vec4(-MaxDepth);
        asdfFragColor.rgba = ivec4(
            float_to_normalized_fixed(-MaxDepth), 
            float_to_normalized_fixed(-MaxDepth),
            float_to_normalized_fixed(-MaxDepth),
            float_to_normalized_fixed(-MaxDepth)
        );
		return;
	}
	
	if (fragDepth > nearestDepth && fragDepth < farthestDepth) {
		// This fragment needs to be peeled again
        if (!gl_FrontFacing) {
            // asdfFragColor.rgba = vec4(-fragDepth, fragDepth, -MaxDepth, -MaxDepth);
            asdfFragColor.rgba = ivec4(
                float_to_normalized_fixed(-fragDepth), 
                float_to_normalized_fixed(fragDepth),
                float_to_normalized_fixed(-MaxDepth),
                float_to_normalized_fixed(-MaxDepth)
            );
        } else {
            // asdfFragColor.rgba = vec4(-MaxDepth, -MaxDepth, -fragDepth, fragDepth);
            asdfFragColor.rgba = ivec4(
                float_to_normalized_fixed(-MaxDepth),
                float_to_normalized_fixed(-MaxDepth),
                float_to_normalized_fixed(-fragDepth), 
                float_to_normalized_fixed(fragDepth)
            );
        }
		return;
	}
	
	// If we made it here, this fragment is on the peeled layer from last pass
	// therefore, we need to shade it, and make sure it is not peeled any farther
	// vec4 color = ShadeFragment();
	// asdfFragColor.rgba = vec4(-MaxDepth);
	
    asdfFragColor.rgba = ivec4(
        float_to_normalized_fixed(-MaxDepth), 
        float_to_normalized_fixed(-MaxDepth),
        float_to_normalized_fixed(-MaxDepth),
        float_to_normalized_fixed(-MaxDepth)
    );
	// if (fragDepth == nearestDepth) {
	// 	gl_FragData[1].xyz += color.rgb * color.a * alphaMultiplier;
	// 	gl_FragData[1].w = 1.0 - alphaMultiplier * (1.0 - color.a);
	// } else {
	// 	gl_FragData[2] += color;
	// }
}
