#include <iostream>

double square(double x)
{
    return x*x;
}

void print_square(double x)
{
    std::cout << "the square of " << x << " is " << square(x) << '\n';
}

int main()
{
    print_square(1.234);
    print_square(5.555);
}