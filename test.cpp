#include <iostream>

int main()
{
int x[3] = {0, 1, 2};
int *p = x;
p = x + 2;
std::cout << p[0]; // you get 2
} 