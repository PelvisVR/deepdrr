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

uniform usampler2DRect DepthBlenderTex;
// uniform samplerRECT FrontBlenderTex;

// #define MAX_DEPTH 10.0
uniform float MaxDepth;

// vec4 ShadeFragment();

uniform vec3 cam_pos;

in vec3 frag_position;

out uvec4 asdfFragColor;

// void main(void)
// {
// 	// gl_asdfFragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
//     float dist = length(frag_position-cam_pos);
//     gl_asdfFragColor.xy = vec2(-dist, dist);
// }


uint float_to_normalized_fixed(float input) {
    // return 8346;
    return 39294u;
    float normalized = input / MaxDepth / 2.0f;
    return uint(clamp(normalized+0.5f, 0.0f, 1.0f) * 4294967295.0f);
}


void main(void)
{
	// window-space depth interpolated linearly in screen space
	// float fragDepth = gl_FragCoord.z;
    float rawDepth = length(frag_position-cam_pos);
    uint fragDepth = float_to_normalized_fixed(rawDepth);
    uint negFragDepth = float_to_normalized_fixed(-rawDepth);


	uvec2 depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).xy;
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

	uint negNearestDepth = -depthBlender.x;
	uint farthestDepth = depthBlender.y;
	// float alphaMultiplier = 1.0 - forwardTemp.w;

    // asdfFragColor.xy = vec2(99999); // TODO
    // return;

	if (negFragDepth >= negNearestDepth || fragDepth >= farthestDepth) {
		// Skip this depth in the peeling algorithm
		// asdfFragColor.rgba = vec4(-MaxDepth);
        asdfFragColor.rgba = uvec4(
            float_to_normalized_fixed(-MaxDepth), 
            float_to_normalized_fixed(-MaxDepth),
            float_to_normalized_fixed(-MaxDepth),
            float_to_normalized_fixed(-MaxDepth)
        );
		return;
	}
	
	if (negFragDepth < negNearestDepth && fragDepth < farthestDepth) {
		// This fragment needs to be peeled again
        if (!gl_FrontFacing) {
            // asdfFragColor.rgba = vec4(-fragDepth, fragDepth, -MaxDepth, -MaxDepth);
            asdfFragColor.rgba = uvec4(
                float_to_normalized_fixed(-fragDepth), 
                float_to_normalized_fixed(fragDepth),
                float_to_normalized_fixed(-MaxDepth),
                float_to_normalized_fixed(-MaxDepth)
            );
        } else {
            // asdfFragColor.rgba = vec4(-MaxDepth, -MaxDepth, -fragDepth, fragDepth);
            asdfFragColor.rgba = uvec4(
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
	
    asdfFragColor.rgba = uvec4(
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
