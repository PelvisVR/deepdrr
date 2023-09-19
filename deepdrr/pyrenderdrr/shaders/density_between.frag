#version 330 core

uniform vec3 cam_pos;
uniform float MaxDepth;
uniform float density;

in vec3 frag_position;

uniform sampler2DRect DepthBlenderTex;


void main(void)
{
    vec2 depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).xy;
    float nearDist = depthBlender.x;
    float farDist = depthBlender.y;
    float facing = !gl_FrontFacing?-1:1;

    float depth = length(frag_position-cam_pos);

    // for each frag closer than neardist
    // += density * facing * -neardist
    // for each frag closer than fardist
    // += density * facing * fardist
    // for each frag between neardist and fardist
    // += density * facing * dist
    float result = 0;

    if (depth < nearDist)
    {
        result += -nearDist;
    }
    
    if (depth < farDist)
    {
        result += farDist;
    }

    if (depth > nearDist && depth < farDist)
    {
        result += depth;
    }




    gl_FragColor.rgba=vec4(result * density * facing,0,0,0);
}
