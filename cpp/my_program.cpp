#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <complex>
#include <cmath>

using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Eigen::VectorXcd;

#define AU_L 5.2917721067e-11 // m
#define AU_T 2.41888432651e-17 // s
#define AU_E 4.35974465e-18 // J
#define EV 1.6021766208e-19 // J
#define AU2ANG (AU_L / 1e-10)
#define AU2EV (AU_E / EV)
#define PI 3.141592653589793238462643383279502884L

double simps(VectorXcd f, int size, double dx)
{
    double so = 0.0, se = 0.0;
    for (int i = 1; i < size - 2; i++)
    {
        if(i % 2 == 1)
        {
            so = so + ((double)real(f[i]));
        }
        else
        {
            se = se + ((double)real(f[i]));
        }
    }
    return dx / 3 * (((double)real(f[0])) + ((double)real(f[size-1])) + 4 * so + 2 * se);
}

int main()
{
    // constantes do problema
    const double E0 = 150.0; // eV
    const double delta_x = 1.0; // angstron
    const double x0 = -20.0; // angstron

    // otimizando
    double L = 100.0; // angstron
    int N = 256;
    double dt = 1e-19; // s

    double L_au = L / AU2ANG;
    double dt_au = dt / AU_T;
    double E0_au = E0 / AU2EV;
    double delta_x_au = delta_x / AU2ANG;
    double x0_au = x0 / AU2ANG;
    double k0_au = sqrt(2.0 * E0_au);

    double dx = L / ((double)(N-1));
    double dx_au = L_au / ((double)(N-1));
    VectorXd x_au(N), x_aux(N);
    for (int i = 0; i < N; i++)
    {
        x_au[i] = -L_au / 2.0 + dx_au * ((double)i);
    }

    // crank-nicolson
    MatrixXcd B(N,N);
    MatrixXcd C(N,N);
    MatrixXcd D(N,N);

    std::complex<double> alpha = - dt_au * (1i / (2.0 * dx_au * dx_au)) / 2.0;
    std::complex<double> beta = 1.0 - dt_au * (- 1i / (dx_au * dx_au)) / 2.0;
    std::complex<double> gamma = 1.0 + dt_au * (- 1i / (dx_au * dx_au)) / 2.0;
    for (int i = 0; i < N; i++)
    {
        if (i > 0)
        {
            B(i,i-1) = alpha;
            C(i,i-1) = -alpha;
        }
        if (i < N - 1) {

            B(i,i+1) = alpha;
            C(i,i+1) = -alpha;
        }
        B(i,i) = beta;
        C(i,i) = gamma;
    }
    D = B.inverse() * C;

    // pacote de onda
    double d2 = pow(delta_x_au, 2);
    double PN = 1.0 / pow(2.0 * PI * d2, 1.0 / 4.0);
    VectorXcd psi(N), psi_aux(N), psi_s(N);
    for (int i = 0; i < N; i++)
    {
        psi[i] = 1i * k0_au * x_au[i] - pow(x_au[i] - x0_au, 2) / (4.0 * d2);
        psi[i] = PN * exp(psi[i]);
    }
    psi_s = psi.conjugate();

    double A0 = simps(psi_s.cwiseProduct(psi), N, dx_au);
    double A = 0.0;
    double xm = x0_au, xm2, xm3;
    double var_norma = 0.0;
    int contador = 0;

    while (xm < -x0_au)
    {
        contador++;
        psi = D * psi;
        psi_s = psi.conjugate();
        A = simps(psi_s.cwiseProduct(psi), N, dx_au);
        xm = simps(psi_s.cwiseProduct(x_au.cwiseProduct(psi)), N, dx_au) / A;
        //std::cout << "x0=" << x0_au * AU2ANG << " & A/A0=" << (100.0 * A / A0) << " & X=" << xm * AU2ANG << " & C=" << contador << "\n";
    }

    var_norma = 100.0 * A / A0;
    x_aux = x_au.cwiseProduct(x_au);
    xm2 = simps(psi_s.cwiseProduct(x_aux.cwiseProduct(psi)), N, dx_au) / A;
    x_aux = x_au.cwiseProduct(x_aux);
    xm3 = simps(psi_s.cwiseProduct(x_aux.cwiseProduct(psi)), N, dx_au) / A;
    double desvpad = sqrt(abs(xm2 - pow(xm, 2)));
    double skewness = (xm3 - 3.0 * xm * pow(desvpad, 2) - pow(xm, 3)) / pow(desvpad, 3);

    std::cout << "L=" << L << " & N=" << N << " & dt=" << dt << " & A/A0=" << var_norma << " & X=" << xm * AU2ANG << " & S=" << desvpad * AU2ANG << " & G=" << skewness << " & C=" << contador << " & T=" << (((double)contador) * dt) << "\n";
}
