///////////////////////////////////////////////////////////////////
/// 
/// 2-Dimensional Euler Equation
/// !!! COMPRESSIBLE !!!
/// 
///////////////////////////////////////////////////////////////////
/// Notes :  
/// 05/25 | Done | Advection equation with HLLC .
/// 05/25 | Done | 2D was wierd that dt would getting small .
///              | MAybe I should give dt constant.
/// 05/26 | Work | Rebuild data reconstruction .
/// 05/29 | Work | PPM or the other methods.
///////////////////////////////////////////////////////////////////

#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <cstdio>
#include <filesystem>
#include <algorithm>

using namespace std;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  GLOBAL  //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Change here to setting in another txt file ???

// Grid set
int         Nx = 512, Ny = 512 , xmax = 10 , ymax = 10;
int         nghost = 2 ;
double      dx = 2*float(xmax)/float(Nx) , dy = 2*float(ymax)/float(Ny);
string      Dir="Data" , prob="Euler_2D_4";

const int   Nx_tot = Nx + 2 * nghost;
const int   Ny_tot = Ny + 2 * nghost;

double      dt=1e-1 , t=0;
int         total_step  = 700;
const int   Estep  = 10;                            // saving data step
double total_calc_time = 0.0;

// Physics constant
const double cs     = 1.0; 
const double cfl    = 0.2;
const double gam    = 5.0/3.0;                      // adaibatic constant
const double G      = 10.0;                         // gravity const
const double inv_dx = 1.0/dx , inv_dy = 1.0/dy;
const double rho_f  = 1e-6   , P_f    = 1e-6;       // Avoid unphysical value 
const double R      = 2.0;                          // initial condition circle radius


////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Grid  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Grid_initail(vector<double>& x , vector<double>& y ){
    
    #   pragma omp parallel for
    for (int i=0 ; i<Nx_tot ; i++){x[i]=-xmax + (i-nghost+0.5)*dx;}    
    #   pragma omp parallel for
    for (int i=0 ; i<Ny_tot ; i++){y[i]=-ymax + (i-nghost+0.5)*dy;}    
}

inline int idx(int i, int j) { return j * Nx_tot + i; }

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Data Struct  /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

//===========================================================================
// Conserved 
struct ConsState {
    double rho ;
    double mu  ;
    double mv  ;
    double E   ;

    // +
    ConsState operator+(const ConsState& other) const {
        return {rho + other.rho ,
                mu  + other.mu  ,
                mv  + other.mv  ,
                E   + other.E   };
    };

    // -
    ConsState operator-(const ConsState& other) const {
        return {rho - other.rho ,
                mu  - other.mu  ,
                mv  - other.mv  ,
                E   - other.E   };
    };

    // *
    ConsState operator*(double scalar) const {
        return {rho * scalar,
                mu  * scalar,
                mv  * scalar, 
                E   * scalar};
    }

    // - Flux
    ConsState operator-() const {
        return {-rho, -mu, -mv, -E};
    }
};

//===========================================================================
// Prime
struct PrimState {
    double rho ;
    double u   ;
    double v   ; 
    double P   ;
};

//===========================================================================
inline PrimState ConsToPrim(const ConsState& U) {
    PrimState P;
    double rho = max(U.rho, rho_f);
    P.rho = rho;
    P.u   = U.mu / rho; 
    P.v   = U.mv / rho;

    double P0 = (gam - 1.0) * (U.E - 0.5 * rho *( P.u * P.u + P.v * P.v));
    P.P   = max(P0, P_f);
    
    return P;
}

//===========================================================================
inline ConsState PrimToCons(const PrimState& P) {
    ConsState C;
    double rho = max(P.rho, rho_f);
    C.rho  = rho;
    C.mu   = P.u * rho; 
    C.mv   = P.v * rho;

    double E = max(P.P , P_f)/(gam - 1.0) + 0.5 * rho *( P.u * P.u + P.v * P.v);
    C.E    = E;
    
    return C;
}

//===========================================================================
using RhsOperator = function<vector<ConsState>(const vector<ConsState>&)>;

//===========================================================================
//////////// Slope limiter ////////////
double slope_Limiter(double a, double b) {

    // minmode
    // if (a * b <= 0.0) return 0.0;
    // return (a > 0.0) ? min(a, b) : max(a, b);
    
    // MC
    if (a * b <= 0.0) return 0.0;
    double c = 2.0 * min(abs(a), abs(b));
    double d = 0.5 * (a + b);
    return (a > 0.0) ? min({c, d}) : max({-c, d});

}

//===========================================================================
// State pressure
inline double Get_Pressure(const ConsState& U){

    double rho = max(U.rho , rho_f);    
    double u   = U.mu / rho;
    double v   = U.mv / rho;
    double P   = (gam - 1.0) * (U.E - 0.5 * rho * (u * u + v * v));
    return max(P , P_f);
}

//===========================================================================
// Flux
inline ConsState Get_Flux(const ConsState& U , char& T_v) {
    double rho = max(U.rho , rho_f);    
    double u   = U.mu / rho;
    double v   = U.mv / rho;
    double P   = Get_Pressure(U);

    ConsState F;
    if (T_v == 'x'){
        F.rho = U.mu;                     
        F.mu  = U.mu * u + P;     
        F.mv  = U.mv * u;    
        F.E   = (U.E + P) * u;   
    }else{
        F.rho = U.mv;                     
        F.mu  = U.mu * v ;     
        F.mv  = U.mv * v +P;    
        F.E   = (U.E + P) * v; 
    }
    return F;
}

//===========================================================================
// C Max
inline double Get_Max_Speed(const ConsState& ULs, const ConsState& URs , char& T_v) {
    double rho_L = max(ULs.rho, rho_f); 
    double rho_R = max(URs.rho, rho_f);

    double c_L = sqrt(gam * max(Get_Pressure(ULs), P_f) / rho_L);
    double c_R = sqrt(gam * max(Get_Pressure(URs), P_f) / rho_R);
    
    // cout << " c_L " << c_L << " , c_R " << c_R << endl;
    if (T_v == 'x') {
        double u_L = ULs.mu / rho_L;
        double u_R = URs.mu / rho_R;
        return max(abs(u_L) + c_L, abs(u_R) + c_R);
    } else {
        double v_L = ULs.mv / rho_L; 
        double v_R = URs.mv / rho_R; 
        return max(abs(v_L) + c_L, abs(v_R) + c_R);
    }
}

//===========================================================================
// dt
double Get_CFL_Dt(const vector<ConsState>& U) {
    double max_speed = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:max_speed)
    for(int j = nghost; j < Ny_tot-nghost; j++) {
        for(int i = nghost; i < Nx_tot-nghost; i++) {
            PrimState p = ConsToPrim(U[idx(i,j)]);
            double a = sqrt(gam * p.P / p.rho);
            double speed_x = abs(p.u) + a;
            double speed_y = abs(p.v) + a;
            max_speed = max(max_speed, max(speed_x, speed_y));
        }
    }
    return cfl / (max_speed * (inv_dx + inv_dy) + 1e-12);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Initial  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void Initial_Value( vector<ConsState>& u0 ,vector<double>& x , vector<double>& y){

    filesystem::path dir_path = "../Data/" + prob ;
    if (!filesystem::exists(dir_path) ) {
        if(filesystem::create_directory(dir_path)){cout << "Success build " << dir_path << endl;}
    };

    // // from nghost to Ni , since the boundary is wrong value.
    #pragma omp parallel for collapse(2) 
    for (int j = nghost ; j < Ny_tot-nghost ; j++){
        for (int i = nghost ; i < Nx_tot-nghost ; i++){
            const double id   = idx(i,j);
            const double rho0 = 5.0;
            const double P0   = 100.0;
            const double U0   = 1.0;

            u0[id].rho =  rho0;
            u0[id].mv  =  rho0 * U0 * sin(x[i]/xmax) * cos(y[i]/ymax);
            u0[id].mv  = -rho0 * U0 * cos(x[i]/xmax) * sin(y[i]/ymax);
            double P   =  P0 + rho0 * U0 * U0 /4.0 * (cos(2.0*x[i]/xmax)+cos(2.0*y[i]/ymax));
            u0[id].E   =  P/(gam - 1.0) + 0.5*(u0[id].mu * u0[id].mu + u0[id].mv * u0[id].mv)/rho0 ;
            // Classical Riemman solver
            // double r   = sqrt(x[i]*x[i] + y[j]*y[j]);
            // if(r < R){
            //     double P  = 100.0 ;
            //     u0[idx].rho = 10;
            //     u0[idx].mu  = 0*u0[idx].rho;
            //     u0[idx].mv  = 0*u0[idx].rho;
            //     // u0[idx(i, j)].E   = P/(gam-1.0) ;
            //     u0[idx].E   = P/(gam - 1.0) + 0.5*(u0[idx].mu * u0[idx].mu + u0[idx].mv * u0[idx].mv)/u0[idx].rho ;
            // }else{
            //     double P  = 0.10 ;
            //     u0[idx].rho = 0.1;
            //     u0[idx].mu  = 0*u0[idx].rho;
            //     u0[idx].mv  = 0*u0[idx].rho;
            //     // u0[idx(i, j)].E   = P/(gam-1.0);
            //     u0[idx].E   = P/(gam - 1.0) + 0.5*(u0[idx].mu * u0[idx].mu + u0[idx].mv * u0[idx].mv)/u0[idx].rho ;
            // }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////
///////////////////////////  Data Reconstruct  //////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void Data_Reconstruct(
    const vector<ConsState>& U,
    vector<ConsState>& UL ,    vector<ConsState>& UR ,   
    vector<ConsState>& UT ,    vector<ConsState>& UB)
{   
    vector<PrimState> P(Nx_tot * Ny_tot);
    
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        P[i] = ConsToPrim(U[i]);
    }

    vector<PrimState> PL(Nx_tot * Ny_tot), PR(Nx_tot * Ny_tot), PT(Nx_tot * Ny_tot), PB(Nx_tot * Ny_tot);
    
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        PL[i] = P[i]; PR[i] = P[i];
        PT[i] = P[i]; PB[i] = P[i];
    }

        // X Reconstruction
        #pragma omp parallel for collapse(2) 
        for (int j = nghost-1; j < Ny_tot-nghost; j++) {
            for (int i = nghost-1; i < Nx_tot-nghost; i++) {
                int id    = idx(i, j);
                int id_p1 = idx(i + 1, j);
                int id_m1 = idx(i - 1, j);
                
                double s_rho = slope_Limiter((P[id].rho - P[id_m1].rho) * inv_dx, (P[id_p1].rho - P[id].rho) * inv_dx);
                double s_u   = slope_Limiter((P[id].u   - P[id_m1].u  ) * inv_dx, (P[id_p1].u   - P[id].u  ) * inv_dx);
                double s_v   = slope_Limiter((P[id].v   - P[id_m1].v  ) * inv_dx, (P[id_p1].v   - P[id].v  ) * inv_dx);
                double s_P   = slope_Limiter((P[id].P   - P[id_m1].P  ) * inv_dx, (P[id_p1].P   - P[id].P  ) * inv_dx);

                PR[id].rho = P[id].rho + 0.5 * dx * s_rho;
                PR[id].u   = P[id].u   + 0.5 * dx * s_u;
                PR[id].v   = P[id].v   + 0.5 * dx * s_v;
                PR[id].P   = P[id].P   + 0.5 * dx * s_P;

                PL[id].rho = P[id].rho - 0.5 * dx * s_rho;
                PL[id].u   = P[id].u   - 0.5 * dx * s_u;
                PL[id].v   = P[id].v   - 0.5 * dx * s_v;
                PL[id].P   = P[id].P   - 0.5 * dx * s_P;
            }
        }

        // Y Reconstruction
        #pragma omp parallel for collapse(2) 
        for (int j = nghost-1; j < Ny_tot-nghost; j++) {
            for (int i = nghost-1; i < Nx_tot-nghost; i++) {
                int id    = idx(i, j);
                int id_p1 = idx(i, j + 1);
                int id_m1 = idx(i, j - 1);
                
                double s_rho = slope_Limiter((P[id].rho - P[id_m1].rho) * inv_dy, (P[id_p1].rho - P[id].rho) * inv_dy);
                double s_u   = slope_Limiter((P[id].u   - P[id_m1].u  ) * inv_dy, (P[id_p1].u   - P[id].u  ) * inv_dy);
                double s_v   = slope_Limiter((P[id].v   - P[id_m1].v  ) * inv_dy, (P[id_p1].v   - P[id].v  ) * inv_dy);
                double s_P   = slope_Limiter((P[id].P   - P[id_m1].P  ) * inv_dy, (P[id_p1].P   - P[id].P  ) * inv_dy);

                PT[id].rho = P[id].rho + 0.5 * dy * s_rho;
                PT[id].u   = P[id].u   + 0.5 * dy * s_u;
                PT[id].v   = P[id].v   + 0.5 * dy * s_v;
                PT[id].P   = P[id].P   + 0.5 * dy * s_P;

                PB[id].rho = P[id].rho - 0.5 * dy * s_rho;
                PB[id].u   = P[id].u   - 0.5 * dy * s_u;
                PB[id].v   = P[id].v   - 0.5 * dy * s_v;
                PB[id].P   = P[id].P   - 0.5 * dy * s_P;
            }
        }
    
    // Convert to ConsState
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        PL[i].rho = max(PL[i].rho, rho_f); PL[i].P = max(PL[i].P, P_f);
        PR[i].rho = max(PR[i].rho, rho_f); PR[i].P = max(PR[i].P, P_f);
        PT[i].rho = max(PT[i].rho, rho_f); PT[i].P = max(PT[i].P, P_f);
        PB[i].rho = max(PB[i].rho, rho_f); PB[i].P = max(PB[i].P, P_f);

        UL[i] = PrimToCons(PL[i]);
        UR[i] = PrimToCons(PR[i]);
        UT[i] = PrimToCons(PT[i]);
        UB[i] = PrimToCons(PB[i]);
    }

}

////////////////////////////////////////////////////////////////////////////
///////////////////////////  Boundary  /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Apply_Boundary(vector<ConsState>& U) {
    #pragma omp parallel for collapse(2) 
    for(int i = 0; i < Nx_tot ; i++){
        for(int g = 0 ; g < nghost ; g++){
            U[idx(i,g)]                    = U[idx(i, nghost)];
            U[idx(i, Ny_tot - nghost + g)] = U[idx(i, Ny_tot - nghost - 1)];
        }
    }

    #pragma omp parallel for collapse(2) 
    for(int j = 0; j < Ny_tot ; j++){
        for(int g = 0 ; g < nghost ; g++){
            U[idx(g,j)]                    = U[idx(nghost,j)];
            U[idx(Nx_tot -  nghost + g,j)] = U[idx(Nx_tot - nghost - 1, j)];
        }
    }

}
//===========================================================================
inline ConsState HLLC_Flux(const ConsState& UL_state, const ConsState& UR_state, char direction) {
    PrimState PL = ConsToPrim(UL_state);
    PrimState PR = ConsToPrim(UR_state);

    double aL = sqrt(gam * PL.P / PL.rho);
    double aR = sqrt(gam * PR.P / PR.rho);

    double uL = (direction == 'x') ? PL.u : PL.v;
    double uR = (direction == 'x') ? PR.u : PR.v;

    double SL = min(uL - aL, uR - aR);
    double SR = max(uL + aL, uR + aR);
    double numerator   = PR.rho * uR * (SR - uR) - PL.rho * uL * (SL - uL) + PL.P - PR.P;
    double denominator = PR.rho * (SR - uR) - PL.rho * (SL - uL);
    double SM = numerator / (denominator + 1e-12);

    if (SL >= 0.0) {
        return Get_Flux(UL_state, direction);
    } 
    else if (SR <= 0.0) {
        return Get_Flux(UR_state, direction);
    } 
    else {
        ConsState fL = Get_Flux(UL_state, direction);
        ConsState fR = Get_Flux(UR_state, direction);
        
        if (SM >= 0.0) {
            ConsState UL_star;
            double factor = PL.rho * (SL - uL) / (SL - SM + rho_f);
            UL_star.rho = factor;
            if (direction == 'x') {
                UL_star.mu = factor * SM;
                UL_star.mv = factor * PL.v;
            } else {
                UL_star.mu = factor * PL.u;
                UL_star.mv = factor * SM;
            }
            UL_star.E = factor * (UL_state.E / PL.rho + (SM - uL) * (SM + PL.P / (PL.rho * (SL - uL) + rho_f)));
            return fL + (UL_star - UL_state) * SL;
        }else{
            ConsState UR_star;
            double factor = PR.rho * (SR - uR) / (SR - SM + rho_f);
            UR_star.rho = factor;
            if (direction == 'x') {
                UR_star.mu = factor * SM;
                UR_star.mv = factor * PR.v;
            } else {
                UR_star.mu = factor * PR.u;
                UR_star.mv = factor * SM;
            }
            UR_star.E = factor * (UR_state.E / PR.rho + (SM - uR) * (SM + PR.P / (PR.rho * (SR - uR) + rho_f)));
            return fR + (UR_star - UR_state) * SR;
        }
    }
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  PDE Operator  ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void Euler_Riemann_Operator(const vector<ConsState>& U , vector<ConsState>& dUdt  ){
    vector<ConsState> UL(Nx_tot * Ny_tot) , UR(Nx_tot * Ny_tot) , UT(Nx_tot * Ny_tot) , UB(Nx_tot * Ny_tot);
    vector<ConsState> Flux_x(Nx_tot * Ny_tot) , Flux_y(Nx_tot * Ny_tot);

double t_start = omp_get_wtime();
    // 1 : data reconstruction | 0.035 s
    Data_Reconstruct(U,UL,UR,UT,UB);
double t_end = omp_get_wtime();
total_calc_time = (t_end - t_start);
cout << " Flux " << total_calc_time << " s" << endl;
    // 2 : Flux 
    
    // X | 0.2 s 
    #pragma omp parallel for collapse(2)
    for (int j = nghost; j < Ny_tot-nghost; j++) {
        for (int i = nghost; i <= Nx_tot-nghost; i++) {
            Flux_x[idx(i, j)] = HLLC_Flux(UR[idx(i-1, j)], UL[idx(i, j)], 'x');
            Flux_y[idx(j, i)] = HLLC_Flux(UT[idx(j, i-1)], UB[idx(j, i)], 'y');
        }
    }
    //===========================================================================
    // Y
    // #pragma omp parallel for collapse(2)
    // for (int j = nghost; j <= Ny_tot-nghost; j++) {
    //     for (int i = nghost; i < Nx_tot-nghost; i++) {
    //             Flux_y[idx(i, j)] = HLLC_Flux(UT[idx(i, j-1)], UB[idx(i, j )], 'y');
    //     }
    // }

    // 3: dU/dt | 0.17 s

    #pragma omp parallel for collapse(2)
    for (int j = nghost; j < Ny_tot-nghost; j++) {
        for (int i = nghost; i < Nx_tot-nghost; i++) {

            ConsState dFx_dx = (Flux_x[idx(i+1, j)] - Flux_x[idx(i, j)]) * inv_dx;
            ConsState dGy_dy = (Flux_y[idx(i, j+1)] - Flux_y[idx(i, j)]) * inv_dy;

            dUdt[idx(i, j)] = -(dFx_dx + dGy_dy);
        }
    }


}    
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  RK4 Solver  //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// RK2
void RK2_Step(const vector<ConsState>& Un, vector<ConsState>& Unext, double dt) {
    vector<ConsState> k1(Nx_tot * Ny_tot );
    vector<ConsState> k2(Nx_tot * Ny_tot);
    vector<ConsState> U_pred(Nx_tot * Ny_tot);
    omp_set_nested(1);
    
    // Riemman Solver 0.08s
    Euler_Riemann_Operator(Un, k1);

    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        U_pred[i] = Un[i] + k1[i] * dt;
    }

    Apply_Boundary(U_pred);

    Euler_Riemann_Operator(U_pred, k2);

    // Floor check 0.001s
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        Unext[i] = Un[i] + (k1[i] + k2[i]) * 0.5 * dt;
        if(Unext[i].rho < 0) {
            Unext[i].rho = rho_f ;
        }
        if(Unext[i].E < 0) {
            Unext[i].E = rho_f ;
        }
    }

    Apply_Boundary(Unext);

}

/////////////////////////////////////////////////////////////////////////////
// Euler
void Euler_Step(
    const vector<ConsState>& Un, 
    vector<ConsState>& Unext,
    vector<ConsState>& k1, 
    double dt) 
{
    Euler_Riemann_Operator(Un, k1);

    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot; i++) {
        Unext[i] = Un[i] + k1[i] * dt;
    }
    
    Apply_Boundary(Unext);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////  Main  ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int main(){
    vector<double> x(Nx_tot) , y(Ny_tot);
    vector<ConsState> u0(Nx_tot * Ny_tot);

    // setting grid
    Grid_initail(x,y);

    // initial value
    Initial_Value(u0, x, y);

    Apply_Boundary(u0);

    vector<ConsState> u_next(Nx_tot * Ny_tot) ;
    for (int step=1 ; step <= total_step ; step++){

        double t_start = omp_get_wtime();
        double dt = Get_CFL_Dt(u0);

        t += dt;

        // Euler_Step(u0, u_next, k1,dt);
        RK2_Step(u0,u_next,dt);

        // check Nan then break
        // auto it = std::find_if(u_next.begin(), u_next.end(), [](const ConsState& s) {
        //     return std::isnan(s.rho) || std::isnan(s.E) || std::isnan(s.mu) || std::isnan(s.mv);
        // });

        // if (it != u_next.end()) {
        //     cout << "  Nan Wrong  " << endl;
        //     break;
        // }

        u0.swap(u_next);

        double t_end = omp_get_wtime();
        total_calc_time += (t_end - t_start);

        // u0=u_next;
        // Apply_Boundary(u0);

        // Output
        // if (step % Estep == 0) {
        //     char buffer[50];
        //     std::snprintf(buffer, sizeof(buffer), "%04d", step / Estep);
        //     std::string stepStr = buffer;

            cout << "==========================================================" << endl;
            cout << " Simulation Time = " << t << " , dt = " << dt << endl;
            cout << "            Step = " << step << " | Center density rho = " << u0[idx(Nx_tot/2,Ny_tot/2)].rho << endl;
            cout << "       Real Time = " << total_calc_time << " s | Spend Time = " << t_end - t_start << " s" << endl;

        //     string Fname = "Final.csv";
        //     ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + stepStr + "_" + Fname);
        //     outFile << "t,x,y,rho,mu,mv,E,u,v,P\n";
        //     for (int i=nghost ; i < Nx+nghost ; i++){
        //         for (int j=nghost ; j < Ny+nghost ; j++){

        //             PrimState P = ConsToPrim(u0[idx(i,j)]);
        //             outFile << t               << " , ";
        //             outFile << x[i]            << " , " << y[j]            << " , " << u0[idx(i,j)].rho  << " , ";
        //             outFile << u0[idx(i,j)].mu << " , " << u0[idx(i,j)].mv << " , " << u0[idx(i,j)].E    << " , "; 
        //             outFile << P.u             << " , " << P.v             << " , " << P.P               << '\n' ;
        //         }
        //     }
        //     outFile.close();
        // }
        // initial value
        // for(int j = nghost ; j < Ny_tot-nghost; j++){
        //     for(int i = nghost ; i < Nx_tot-nghost ; i++){
        //         cout << setw(4) << setprecision(2) << u0[idx(i,j)].rho << " , " ;
        //     }
        //     cout << endl;
        // }

        // for(int i=nghost;i<Nx_tot-nghost;i++){cout << "(" << i << " , " << x[i] << ")" <<endl;}
    }

}