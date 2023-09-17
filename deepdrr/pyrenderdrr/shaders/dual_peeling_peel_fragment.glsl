//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

// #extension ARB_draw_buffers : require
#version 330

uniform sampler2DRect DepthBlenderTex;
// uniform samplerRECT FrontBlenderTex;

// #define MAX_DEPTH 10.0
uniform float MaxDepth;
// uniform bool gl_FrontFacing;

// vec4 ShadeFragment();

uniform vec3 cam_pos;
uniform int mode;

in vec3 frag_position;

out vec2 frag_color;

// void main(void)
// {
// 	// gl_FragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
//     float dist = length(frag_position-cam_pos);
//     gl_FragColor.xy = vec2(-dist, dist);
// }


void main(void)
{
	// window-space depth interpolated linearly in screen space
	// float fragDepth = gl_FragCoord.z;
    float fragDepth = length(frag_position-cam_pos);
    // write to depth buffer


	float depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).y;
	// vec2 depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).xy;
	// vec2 depthBlender = textureRect(DepthBlenderTex, gl_FragCoord.xy).xy;
	// vec4 forwardTemp = textureRect(FrontBlenderTex, gl_FragCoord.xy);
	
	// Depths and 1.0-alphaMult always increase
	// so we can use pass-through by default with MAX blending
	// gl_FragData[0].xy = depthBlender;
	
	// // Front colors always increase (DST += SRC*ALPHA_MULT)
	// // so we can use pass-through by default with MAX blending
	// gl_FragData[1] = forwardTemp;
	
	// // Because over blending makes color increase or decrease,
	// // we cannot pass-through by default.
	// // Each pass, only one fragment writes a color greater than 0
	// gl_FragData[2] = vec4(0.0);

	// float nearestDepth = -depthBlender.x;
	// float farthestDepth = depthBlender.y;
	// float alphaMultiplier = 1.0 - forwardTemp.w;

    // gl_FragData[0].xy = vec2(99999); // TODO
    // return;

	// if (fragDepth < nearestDepth || fragDepth > farthestDepth) {
	// 	// // Skip this depth in the peeling algorithm
	// 	// gl_FragData[0].rgba = vec4(-MaxDepth);
	// 	// return;
    //     discard;
	// }

    
	// if (fragDepth > nearestDepth && fragDepth < farthestDepth) {
	// 	// This fragment needs to be peeled again
    //     if (!gl_FrontFacing) {
    //         gl_FragData[0].rgba = vec4(-fragDepth, fragDepth, -MaxDepth, -MaxDepth);
    //     } else {
    //         gl_FragData[0].rgba = vec4(-MaxDepth, -MaxDepth, -fragDepth, fragDepth);
    //     }
	// 	return;
	// }

    // frag_color.rg = vec2(intBitsToFloat(1000), texture2DRect(DepthBlenderTex, gl_FragCoord.xy).y);
    // frag_color.rg = uvec2(1148846080, floatBitsToUint(depthBlender));

    // frag_color.rg = vec2(intBitsToFloat(0), depthBlender);
    // gl_FragDepth = fragDepth / MaxDepth;
    
    if (mode == 0) {
        if (fragDepth <= depthBlender) {
            frag_color.rg = vec2(intBitsToFloat(314159), MaxDepth);
            gl_FragDepth = 1f;

        } else {
            frag_color.rg = vec2(intBitsToFloat(0), fragDepth);
            gl_FragDepth = fragDepth / MaxDepth;
        }
    } else {
        if (fragDepth >= depthBlender) {
            frag_color.rg = vec2(intBitsToFloat(0), 0);
            gl_FragDepth = 0f;
        } else {
            frag_color.rg = vec2(intBitsToFloat(314159), fragDepth);
            gl_FragDepth = fragDepth / MaxDepth;
        }
    }
	
	
	// If we made it here, this fragment is on the peeled layer from last pass
	// therefore, we need to shade it, and make sure it is not peeled any farther
	// vec4 color = ShadeFragment();
	// gl_FragData[0].rgba = vec4(-MaxDepth);
	
	// if (fragDepth == nearestDepth) {
	// 	gl_FragData[1].xyz += color.rgb * color.a * alphaMultiplier;
	// 	gl_FragData[1].w = 1.0 - alphaMultiplier * (1.0 - color.a);
	// } else {
	// 	gl_FragData[2] += color;
	// }
}
