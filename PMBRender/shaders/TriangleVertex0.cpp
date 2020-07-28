#version 450 core
layout(std140, row_major, binding = 0) uniform transBuffer
{
	mat4 trans;
};
layout(std140, binding = 1) uniform  Light
{
	vec3 light;
};
//layout(std430, binding = 2) buffer Normal
//{
//	vec3 normal[];
//};
layout(location = 1) in vec3 position;
out vec4 in_color;
void main()
{
	gl_Position = trans * vec4(position, 1);
	//float k = max(dot(light.xyz, normal[gl_VertexID / 3]), 0);
	in_color =
		vec4(
			0.5 * (atan(sin(position.x * 100)) + 1),
			0.5 * (atan(sin(position.y * 100)) + 1),
			0.5 * (atan(sin(position.z * 100)) + 1), 1);
	//0.5 * (atan(position.x) + 1),
	//	0.5 * (atan(position.y) + 1),
	//	0.5 * (atan(position.z) + 1), 1);
}