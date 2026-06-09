///////////////////////////////////////////////////////////////////
/// 
/// 3-dimensional Euler Equations
/// Author : TsungHsuan Chen - NTNU
///================================================================
/// Notes :  
/// 05/17 | Done | First try 3D Euler simulation.
///              | 
///
///================================================================
/// Reference : 
///     
/// 
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
// namespace fs = std::filesystem;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  GLOBAL  //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Change here to setting in another txt file ???

const int     Nx = 64*1 , Ny = 64*1 , Nz = 64;
const double  xmax = 2.0 , ymax = 2.0 ,zmax = 1.0;
const double  dx = 2*float(xmax)/Nx , dy = 2*float(ymax)/Ny , dz = 2*float(zmax)/Nz ;
const int     nghost = 3 ;
const string  Dir="Data" , prob = "Euler_3D_KHI_PPM_v";


const int   Nx_tot = Nx + 2 * nghost;
const int   Ny_tot = Ny + 2 * nghost;
const int   Nz_tot = Nz + 2 * nghost;

double      dt=1e-1 , t=0;
int         total_step  = 2000;
const int   Estep  = 20;                            // saving data step
double total_calc_time = 0.0;

// Physics constant
const double cs     = 1.0; 
const double cfl    = 0.5;
const double gam    = 5.0/3.0;                      // adaibatic constant
const double G      = 10.0;                         // gravity const
const double inv_dx = 1.0/dx , inv_dy = 1.0/dy , inv_dz = 1.0/dz ;
const double rho_f  = 1e-6   , P_f    = 1e-6;       // Avoid unphysical value 
const double R      = 0.2;                          // initial condition circle radius

////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Grid  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Grid_initail(vector<double>& x , vector<double>& y , vector<double>& z ){
    
    #   pragma omp parallel for
    for (int i=0 ; i<Nx_tot ; i++){x[i]=-xmax + (i-nghost+0.5)*dx;}    
    #   pragma omp parallel for
    for (int j=0 ; j<Ny_tot ; j++){y[j]=-ymax + (j-nghost+0.5)*dy;}    
    #   pragma omp parallel for
    for (int k=0 ; k<Nz_tot ; k++){z[k]=-zmax + (k-nghost+0.5)*dz;}    
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Data Struct  /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
struct ConsState {
    double rho;  // Dsensity
    double mu ;  // x momenta rho * u
    double mv ;  // y momenta rho * v
    double mw ;  // z momenta rho * w
    double E  ;

    // +
    ConsState operator+(const ConsState& other) const {
        return {rho + other.rho ,
                mu  + other.mu  ,
                mv  + other.mv  ,
                mw  + other.mw  , 
                E   + other.E  };
    };
    
    // -
    ConsState operator-(const ConsState& other) const {
        return {rho - other.rho ,
                mu  - other.mu  ,
                mv  - other.mv  ,
                mw  - other.mw  , 
                E   - other.E  };
    };

    // *
    ConsState operator*(double scalar) const {
        return {rho * scalar  ,
                mu  * scalar  ,
                mv  * scalar  , 
                mw  * scalar  , 
                E   * scalar };
    }
    
    // - Flux
    ConsState operator-() const {
        return {-rho, -mu, -mv, -mw ,-E};
    }

};

inline double S_rho(double rho) { return (rho < 1e-10) ? 1e-10 : rho; }

// inline int idx(int i, int j , int k)    { return k * Ny *Nx + j * Nx + i; }
inline int idx(int i, int j, int k) { 
    return k * Ny_tot * Nx_tot + j * Nx_tot + i; 
};
//===========================================================================
// Prime
struct PrimState {
    double rho ;
    double u   ;
    double v   ; 
    double w   ; 
    double P   ;
};

//===========================================================================
inline PrimState ConsToPrim(const ConsState& U) {
    PrimState P;
    double rho = max(U.rho, rho_f);
    P.rho = rho;
    P.u   = U.mu / rho; 
    P.v   = U.mv / rho;
    P.w   = U.mw / rho;

    double P0 = (gam - 1.0) * (U.E - 0.5 * rho *( P.u * P.u + P.v * P.v + P.w * P.w));
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
    C.mw   = P.w * rho;

    double E = max(P.P , P_f)/(gam - 1.0) + 0.5 * rho *( P.u * P.u + P.v * P.v+ P.w * P.w);
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
    double w   = U.mw / rho;
    double P   = (gam - 1.0) * (U.E - 0.5 * rho * (u * u + v * v + w * w));
    return max(P , P_f);
}

//===========================================================================
// Flux
inline ConsState Get_Flux(const ConsState& U , char T_v) {
    double rho = max(U.rho , rho_f);    
    double u   = U.mu / rho;
    double v   = U.mv / rho;
    double w   = U.mw / rho;
    double P   = Get_Pressure(U);

    ConsState F;
    if (T_v == 'x'){
        F.rho = U.mu            ;                     
        F.mu  = U.mu * u + P    ;     
        F.mv  = U.mv * u        ;    
        F.mw  = U.mw * u        ;    
        F.E   = (U.E + P) * u   ;   
    }else if (T_v == 'y'){
        F.rho = U.mv            ;                     
        F.mu  = U.mu * v        ;     
        F.mv  = U.mv * v + P    ;    
        F.mw  = U.mw * v        ;    
        F.E   = (U.E + P) * v   ; 
    }else{
        F.rho = U.mw            ;                     
        F.mu  = U.mu * w        ;     
        F.mv  = U.mv * w        ;    
        F.mw  = U.mw * w + P    ;    
        F.E   = (U.E + P) * w   ; 
    }
    return F;
}
//===========================================================================
// C Max
inline double Get_Max_Speed(const ConsState& ULs, const ConsState& URs , char T_v) {
    double rho_L = max(ULs.rho, rho_f); 
    double rho_R = max(URs.rho, rho_f);

    double c_L = sqrt(gam * max(Get_Pressure(ULs), P_f) / rho_L);
    double c_R = sqrt(gam * max(Get_Pressure(URs), P_f) / rho_R);
    
    // cout << " c_L " << c_L << " , c_R " << c_R << endl;
    if (T_v == 'x') {
        double u_L = ULs.mu / rho_L;
        double u_R = URs.mu / rho_R;
        return max(abs(u_L) + c_L, abs(u_R) + c_R);
    } else if (T_v == 'y'){
        double v_L = ULs.mv / rho_L; 
        double v_R = URs.mv / rho_R; 
        return max(abs(v_L) + c_L, abs(v_R) + c_R);
    }else{
        double v_L = ULs.mw / rho_L; 
        double v_R = URs.mw / rho_R; 
        return max(abs(v_L) + c_L, abs(v_R) + c_R);
    }
}
//===========================================================================
// dt
double Get_CFL_Dt(const vector<ConsState>& U) {
    double max_speed = 0.0;

    #pragma omp parallel for collapse(3) reduction(max:max_speed)
    for(int k = nghost; k < Nz_tot-nghost; k++) {
        for(int j = nghost; j < Ny_tot-nghost; j++) {
            for(int i = nghost; i < Nx_tot-nghost; i++) {
                PrimState p = ConsToPrim(U[idx(i,j,k)]);
                double a = sqrt(gam * p.P / p.rho);
                // double speed_x = abs(p.u) + a;
                // double speed_y = abs(p.v) + a;
                // double speed_z = abs(p.w) + a;
                double local_cfl = (abs(p.u)+a)*inv_dx + (abs(p.v)+a)*inv_dy + (abs(p.w)+a)*inv_dz;
                max_speed = max(max_speed, local_cfl );

            }
        }
    }
    // return cfl / (max_speed * (inv_dx + inv_dy +inv_dz) + 1e-12);
    return cfl / (max_speed + 1e-12);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Initial  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Initial_Value( vector<ConsState>& u0 ,vector<double>& x , vector<double>& y , vector<double>& z){
    // V1 - Testing struct
    filesystem::path dir_path = "../Data/" + prob ;
    if (!filesystem::exists(dir_path) ) {
        if(filesystem::create_directory(dir_path)){cout << "Success build " << dir_path << endl;}
    };

    // #   pragma omp parallel for collapse(3)
    // for (int k=0 ; k<Nz_tot ; k++){
    //     for (int j=0 ; j<Ny_tot ; j++){
    //         for (int i=0 ; i<Nx_tot ; i++){
    //             const int id = idx(i, j ,k);

    //             // Initail condiction
    //             double r = (x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
    //             if(r < R * R){
    //                 double P  = 100.0 ;
    //                 double mu=0 , mv= 0;
    //                 u0[id].rho = 10;
    //                 u0[id].mu  = 0*u0[id].rho;
    //                 u0[id].mv  = 0*u0[id].rho;
    //                 u0[id].mw  = 0*u0[id].rho;
    //                 // u0[id(i, j)].E   = P/(gam-1.0) ;
    //                 u0[id].E   = 10*P/(gam - 1.0) + 0.5*(u0[id].mu * u0[id].mu + u0[id].mv * u0[id].mv + u0[id].mw * u0[id].mw)/u0[id].rho ;
    //             }else{
    //                 double P  = 0.10 ;
    //                 u0[id].rho = 0.1;
    //                 u0[id].mu  = 0*u0[id].rho;
    //                 u0[id].mv  = 0*u0[id].rho;
    //                 u0[id].mw  = 0*u0[id].rho;
    //                 // u0[id(i, j)].E   = P/(gam-1.0);
    //                 u0[id].E   = P/(gam - 1.0) + 0.5*(u0[id].mu * u0[id].mu + u0[id].mv * u0[id].mv+ u0[id].mw * u0[id].mw)/u0[id].rho ;
    //             }
    //         }
    //     }    
    // }
    
    #   pragma omp parallel for collapse(3)
    for (int k=0 ; k<Nz_tot ; k++){
        for (int j=0 ; j<Ny_tot ; j++){
            for (int i=0 ; i<Nx_tot ; i++){      
            const int id = idx(i, j ,k );
            double rho, u, v, w, P;
            double ang = 45;
            double rad = ang * M_PI / 180.0;

            // P = 2.5;
            // rho= 10;
            // u = 0;
            // v = 0;
            
            // if (x[i]*x[i]+y[j]*y[j] <= 0.2) { 
            //     w = 2;
            // }else{
            //     w=0;
            // }

            P  = 2.5; 
            const double v0 = 1.0;

            if (abs(z[k]) < 0.25) {
                rho =  2.0 ;       
                u   =  -v0 ;
                v   =  0.0 ;
            } else {
                rho =  1.0 ;             
                u   =  v0  ;
                v   =  0   ;
                // v   =  v0 * cos(rad) ;
                // u   = v0 * cos(rad);     
                // v   = v0 * sin(rad); 
            }

            double perturbation = 0.1 * sin(2.0 * M_PI * x[i] / (2.0 * xmax));
            w = perturbation;
            
            u0[id].rho = rho;
            u0[id].mu  = rho * u;
            u0[id].mv  = rho * v;
            u0[id].mw  = rho * w;
            u0[id].E   = P / (gam - 1.0) + 0.5 * rho * (u * u + v * v + w * w);

            }
        }
    }

    char buffer[50];
    std::snprintf(buffer, sizeof(buffer), "%04d", 0);
    std::string stepStr = buffer;
    string Fname = "Final.csv";

    ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + stepStr + "_" + Fname);
    outFile << "time,x,y,z,rho,u,v,w,E,P\n";

    for(int k = nghost; k < Nz_tot-nghost; k++) {
        for(int j = nghost; j < Ny_tot-nghost; j++) {
            for(int i = nghost; i < Nx_tot-nghost; i++) {
                double id = idx(i,j,k);
                PrimState P = ConsToPrim(u0[id]);

                outFile << t            << " , ";
                outFile << x[i]         << " , " << y[j]  << " , " << z[k]   << " , " << u0[id].rho  << " , ";
                outFile << P.u          << " , " << P.v   << " , " << P.w    << " , " ;
                outFile << u0[id].E     << " , " << P.P ;
                outFile << '\n';
            }
        }
    }
    outFile.close();

}

/////////////////////////////////////////////////////////////////////////////
///////////////////////////  Data Reconstruct  //////////////////////////////
/////////////////////////////////////////////////////////////////////////////
vector<PrimState> P(Nx_tot * Ny_tot * Nz_tot);
// vector<PrimState> PL(Nx_tot * Ny_tot * Nz_tot) , PR(Nx_tot * Ny_tot * Nz_tot) ;
// vector<PrimState> PT(Nx_tot * Ny_tot * Nz_tot) , PB(Nx_tot * Ny_tot * Nz_tot) ;
// vector<PrimState> PF(Nx_tot * Ny_tot * Nz_tot) , PBa(Nx_tot * Ny_tot * Nz_tot) ;

void Data_Reconstruct_PLM(
    const vector<ConsState>& U,
    vector<ConsState>& UL ,    vector<ConsState>& UR ,   
    vector<ConsState>& UT ,    vector<ConsState>& UB ,
    vector<ConsState>& UF ,    vector<ConsState>& UBa)
{   
    
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot * Nz_tot; i++) {
        P[i] = ConsToPrim(U[i]);
    }
    
    // PLM
    // X Reconstruction
    #pragma omp parallel for collapse(3) 
    for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
        for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
            for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
                int id    = idx(i, j, k);
                int id_p1 = idx(i + 1, j, k);
                int id_m1 = idx(i - 1, j, k);
                
                // Make it doing struct operator (s.rho ... )?
                double s_rho = slope_Limiter((P[id].rho - P[id_m1].rho) * inv_dx, (P[id_p1].rho - P[id].rho) * inv_dx);
                double s_u   = slope_Limiter((P[id].u   - P[id_m1].u  ) * inv_dx, (P[id_p1].u   - P[id].u  ) * inv_dx);
                double s_v   = slope_Limiter((P[id].v   - P[id_m1].v  ) * inv_dx, (P[id_p1].v   - P[id].v  ) * inv_dx);
                double s_w   = slope_Limiter((P[id].w   - P[id_m1].w  ) * inv_dx, (P[id_p1].w   - P[id].w  ) * inv_dx);
                double s_P   = slope_Limiter((P[id].P   - P[id_m1].P  ) * inv_dx, (P[id_p1].P   - P[id].P  ) * inv_dx);

                PrimState pr, pl;

                pr.rho = P[id].rho + 0.5 * dx * s_rho; 
                pr.u   = P[id].u   + 0.5 * dx * s_u;
                pr.v   = P[id].v   + 0.5 * dx * s_v;
                pr.w   = P[id].w   + 0.5 * dx * s_w;
                pr.P   = P[id].P   + 0.5 * dx * s_P;
                    
                pl.rho = P[id].rho - 0.5 * dx * s_rho;
                pl.u   = P[id].u   - 0.5 * dx * s_u;
                pl.v   = P[id].v   - 0.5 * dx * s_v;
                pl.w   = P[id].w   - 0.5 * dx * s_w;
                pl.P   = P[id].P   - 0.5 * dx * s_P;
                
                pr.rho = max(pr.rho, rho_f); pr.P = max(pr.P, P_f);
                pl.rho = max(pl.rho, rho_f); pl.P = max(pl.P, P_f);

                UR[id] = PrimToCons(pr);
                UL[id] = PrimToCons(pl);
            }
        }
    }

    // Y Reconstruction
    #pragma omp parallel for collapse(3) 
    for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
        for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
            for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
                int id    = idx(i, j, k);
                int id_p1 = idx(i, j + 1, k);
                int id_m1 = idx(i, j - 1, k);
                
                double s_rho = slope_Limiter((P[id].rho - P[id_m1].rho) * inv_dy, (P[id_p1].rho - P[id].rho) * inv_dy);
                double s_u   = slope_Limiter((P[id].u   - P[id_m1].u  ) * inv_dy, (P[id_p1].u   - P[id].u  ) * inv_dy);
                double s_v   = slope_Limiter((P[id].v   - P[id_m1].v  ) * inv_dy, (P[id_p1].v   - P[id].v  ) * inv_dy);
                double s_w   = slope_Limiter((P[id].w   - P[id_m1].w  ) * inv_dy, (P[id_p1].w   - P[id].w  ) * inv_dy);
                double s_P   = slope_Limiter((P[id].P   - P[id_m1].P  ) * inv_dy, (P[id_p1].P   - P[id].P  ) * inv_dy);

                PrimState pt, pb;
                pt.rho = P[id].rho + 0.5 * dy * s_rho; 
                pt.u   = P[id].u   + 0.5 * dy * s_u;
                pt.v   = P[id].v   + 0.5 * dy * s_v;
                pt.w   = P[id].w   + 0.5 * dy * s_w;
                pt.P   = P[id].P   + 0.5 * dy * s_P;
                    
                pb.rho = P[id].rho - 0.5 * dy * s_rho;
                pb.u   = P[id].u   - 0.5 * dy * s_u;
                pb.v   = P[id].v   - 0.5 * dy * s_v;
                pb.w   = P[id].w   - 0.5 * dy * s_w;
                pb.P   = P[id].P   - 0.5 * dy * s_P;

                pt.rho = max(pt.rho, rho_f); pt.P = max(pt.P, P_f);
                pb.rho = max(pb.rho, rho_f); pb.P = max(pb.P, P_f);
                
                UT[id] = PrimToCons(pt);
                UB[id] = PrimToCons(pb);
            }
        }
    }

    // Z Reconstruction
    #pragma omp parallel for collapse(3) 
    for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
        for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
            for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
                int id    = idx(i, j , k);
                int id_p1 = idx(i, j , k + 1);
                int id_m1 = idx(i, j , k - 1);
                
                double s_rho = slope_Limiter((P[id].rho - P[id_m1].rho) * inv_dz, (P[id_p1].rho - P[id].rho) * inv_dz);
                double s_u   = slope_Limiter((P[id].u   - P[id_m1].u  ) * inv_dz, (P[id_p1].u   - P[id].u  ) * inv_dz);
                double s_v   = slope_Limiter((P[id].v   - P[id_m1].v  ) * inv_dz, (P[id_p1].v   - P[id].v  ) * inv_dz);
                double s_w   = slope_Limiter((P[id].w   - P[id_m1].w  ) * inv_dz, (P[id_p1].w   - P[id].w  ) * inv_dz);
                double s_P   = slope_Limiter((P[id].P   - P[id_m1].P  ) * inv_dz, (P[id_p1].P   - P[id].P  ) * inv_dz);

                PrimState pt, pb;
                pt.rho = P[id].rho + 0.5 * dz * s_rho; 
                pt.u   = P[id].u   + 0.5 * dz * s_u;
                pt.v   = P[id].v   + 0.5 * dz * s_v;
                pt.w   = P[id].w   + 0.5 * dz * s_w;
                pt.P   = P[id].P   + 0.5 * dz * s_P;
                    
                pb.rho = P[id].rho - 0.5 * dz * s_rho;
                pb.u   = P[id].u   - 0.5 * dz * s_u;
                pb.v   = P[id].v   - 0.5 * dz * s_v;
                pb.w   = P[id].w   - 0.5 * dz * s_w;
                pb.P   = P[id].P   - 0.5 * dz * s_P;

                pt.rho = max(pt.rho, rho_f); pt.P = max(pt.P, P_f);
                pb.rho = max(pb.rho, rho_f); pb.P = max(pb.P, P_f);
                
                UF[id] = PrimToCons(pt);
                UBa[id] = PrimToCons(pb);
            }
        }
    }
}

//===========================================================================
// PPM

auto monetize = [](double L, double R, double C) -> pair<double, double> {
            // local maxinmum , Const
            if ((R - C) * (C - L) <= 0.0) {
                L = C; R = C;
            } else {
                double dP = R - L;
                double P6 = 6.0 * (C - 0.5 * (L + R));
                // Over
                if (dP * P6 > dP * dP) {
                    L = 3.0 * C - 2.0 * R;
                } else if (dP * P6 < -dP * dP) {
                    R = 3.0 * C - 2.0 * L;
                }
            }
            return {L, R};
};

// void Data_Reconstruct_PPM(    
//     const vector<ConsState>& U,
//     vector<ConsState>& UL ,    vector<ConsState>& UR ,   
//     vector<ConsState>& UT ,    vector<ConsState>& UB ,
//     vector<ConsState>& UF ,    vector<ConsState>& UBa)
// {   
//     #pragma omp parallel for
//     for (int i = 0; i < Nx_tot * Ny_tot * Nz_tot; i++) {
//         P[i] = ConsToPrim(U[i]);
//     }

//     // X Reconstruction
//     #pragma omp parallel for collapse(3) 
//     for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
//         for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
//             for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
//                 int id    = idx(i, j, k);
//                 int id_p1 = idx(i + 1, j, k);
//                 int id_p2 = idx(i + 2, j, k);
//                 int id_m1 = idx(i - 1, j, k);
//                 int id_m2 = idx(i - 2, j, k);

//                 double P_r_rho = (7.0/12.0)*(P[id].rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
//                 double P_r_u   = (7.0/12.0)*(P[id].u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
//                 double P_r_v   = (7.0/12.0)*(P[id].v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
//                 double P_r_w   = (7.0/12.0)*(P[id].w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
//                 double P_r_P   = (7.0/12.0)*(P[id].P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

//                 double P_l_rho = (7.0/12.0)*(P[id_m1].rho + P[id].rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
//                 double P_l_u   = (7.0/12.0)*(P[id_m1].u   + P[id].u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
//                 double P_l_v   = (7.0/12.0)*(P[id_m1].v   + P[id].v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
//                 double P_l_w   = (7.0/12.0)*(P[id_m1].w   + P[id].w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
//                 double P_l_P   = (7.0/12.0)*(P[id_m1].P   + P[id].P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

//                 auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, P[id].rho);
//                 auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   P[id].u);
//                 auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   P[id].v);
//                 auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   P[id].w);
//                 auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   P[id].P);
                
//                 UL[id] = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
//                 UR[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
//             }
//         }
//     }

//     // Y Reconstruction
//     #pragma omp parallel for collapse(3) 
//     for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
//         for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
//             for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
//                 int id    = idx(i, j, k);
//                 int id_p1 = idx(i , j + 1, k);
//                 int id_p2 = idx(i , j + 2, k);
//                 int id_m1 = idx(i , j - 1, k);
//                 int id_m2 = idx(i , j - 2, k);

//                 double P_r_rho = (7.0/12.0)*(P[id].rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
//                 double P_r_u   = (7.0/12.0)*(P[id].u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
//                 double P_r_v   = (7.0/12.0)*(P[id].v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
//                 double P_r_w   = (7.0/12.0)*(P[id].w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
//                 double P_r_P   = (7.0/12.0)*(P[id].P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

//                 double P_l_rho = (7.0/12.0)*(P[id_m1].rho + P[id].rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
//                 double P_l_u   = (7.0/12.0)*(P[id_m1].u   + P[id].u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
//                 double P_l_v   = (7.0/12.0)*(P[id_m1].v   + P[id].v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
//                 double P_l_w   = (7.0/12.0)*(P[id_m1].w   + P[id].w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
//                 double P_l_P   = (7.0/12.0)*(P[id_m1].P   + P[id].P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

//                 auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, P[id].rho);
//                 auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   P[id].u);
//                 auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   P[id].v);
//                 auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   P[id].w);
//                 auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   P[id].P);
                
//                 UT[id] = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
//                 UB[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
//             }
//         }
//     }

//     // Z Reconstruction
//     #pragma omp parallel for collapse(3) 
//     for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
//         for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
//             for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
//                 int id    = idx(i, j, k);
//                 int id_p1 = idx(i , j , k + 1);
//                 int id_p2 = idx(i , j , k + 2);
//                 int id_m1 = idx(i , j , k - 1);
//                 int id_m2 = idx(i , j , k - 2);

//                 double P_r_rho = (7.0/12.0)*(P[id].rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
//                 double P_r_u   = (7.0/12.0)*(P[id].u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
//                 double P_r_v   = (7.0/12.0)*(P[id].v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
//                 double P_r_w   = (7.0/12.0)*(P[id].w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
//                 double P_r_P   = (7.0/12.0)*(P[id].P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

//                 double P_l_rho = (7.0/12.0)*(P[id_m1].rho + P[id].rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
//                 double P_l_u   = (7.0/12.0)*(P[id_m1].u   + P[id].u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
//                 double P_l_v   = (7.0/12.0)*(P[id_m1].v   + P[id].v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
//                 double P_l_w   = (7.0/12.0)*(P[id_m1].w   + P[id].w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
//                 double P_l_P   = (7.0/12.0)*(P[id_m1].P   + P[id].P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

//                 auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, P[id].rho);
//                 auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   P[id].u);
//                 auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   P[id].v);
//                 auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   P[id].w);
//                 auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   P[id].P);
                
//                 UF[id]  = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
//                 UBa[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
//             }
//         }
//     }

// }

void Data_Reconstruct_PPM(    
    const vector<ConsState>& U,
    vector<ConsState>& UL ,    vector<ConsState>& UR ,   
    vector<ConsState>& UT ,    vector<ConsState>& UB ,
    vector<ConsState>& UF ,    vector<ConsState>& UBa)
{   
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot * Nz_tot; i++) {
        P[i] = ConsToPrim(U[i]);
    }

    #pragma omp parallel for collapse(3) 
    for (int k = nghost-1; k <= Nz_tot-nghost; k++) {
        for (int j = nghost-1; j <= Ny_tot-nghost; j++) {
            for (int i = nghost-1; i <= Nx_tot-nghost; i++) {
                
                int id = idx(i, j, k);
                PrimState p_center = P[id]; 

                {
                    int id_p1 = idx(i + 1, j, k);
                    int id_p2 = idx(i + 2, j, k);
                    int id_m1 = idx(i - 1, j, k);
                    int id_m2 = idx(i - 2, j, k);

                    double P_r_rho = (7.0/12.0)*(p_center.rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
                    double P_r_u   = (7.0/12.0)*(p_center.u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
                    double P_r_v   = (7.0/12.0)*(p_center.v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
                    double P_r_w   = (7.0/12.0)*(p_center.w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
                    double P_r_P   = (7.0/12.0)*(p_center.P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

                    double P_l_rho = (7.0/12.0)*(P[id_m1].rho + p_center.rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
                    double P_l_u   = (7.0/12.0)*(P[id_m1].u   + p_center.u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
                    double P_l_v   = (7.0/12.0)*(P[id_m1].v   + p_center.v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
                    double P_l_w   = (7.0/12.0)*(P[id_m1].w   + p_center.w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
                    double P_l_P   = (7.0/12.0)*(P[id_m1].P   + p_center.P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

                    auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, p_center.rho);
                    auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   p_center.u);
                    auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   p_center.v);
                    auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   p_center.w);
                    auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   p_center.P);
                    
                    UL[id] = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
                    UR[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
                }

                {
                    int id_p1 = idx(i, j + 1, k);
                    int id_p2 = idx(i, j + 2, k);
                    int id_m1 = idx(i, j - 1, k);
                    int id_m2 = idx(i, j - 2, k);

                    double P_r_rho = (7.0/12.0)*(p_center.rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
                    double P_r_u   = (7.0/12.0)*(p_center.u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
                    double P_r_v   = (7.0/12.0)*(p_center.v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
                    double P_r_w   = (7.0/12.0)*(p_center.w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
                    double P_r_P   = (7.0/12.0)*(p_center.P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

                    double P_l_rho = (7.0/12.0)*(P[id_m1].rho + p_center.rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
                    double P_l_u   = (7.0/12.0)*(P[id_m1].u   + p_center.u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
                    double P_l_v   = (7.0/12.0)*(P[id_m1].v   + p_center.v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
                    double P_l_w   = (7.0/12.0)*(P[id_m1].w   + p_center.w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
                    double P_l_P   = (7.0/12.0)*(P[id_m1].P   + p_center.P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

                    auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, p_center.rho);
                    auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   p_center.u);
                    auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   p_center.v);
                    auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   p_center.w);
                    auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   p_center.P);
                    
                    UT[id] = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
                    UB[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
                }

                {
                    int id_p1 = idx(i, j, k + 1);
                    int id_p2 = idx(i, j, k + 2);
                    int id_m1 = idx(i, j, k - 1);
                    int id_m2 = idx(i, j, k - 2);

                    double P_r_rho = (7.0/12.0)*(p_center.rho + P[id_p1].rho) - (1.0/12.0)*(P[id_m1].rho + P[id_p2].rho);
                    double P_r_u   = (7.0/12.0)*(p_center.u   + P[id_p1].u  ) - (1.0/12.0)*(P[id_m1].u   + P[id_p2].u  );
                    double P_r_v   = (7.0/12.0)*(p_center.v   + P[id_p1].v  ) - (1.0/12.0)*(P[id_m1].v   + P[id_p2].v  );
                    double P_r_w   = (7.0/12.0)*(p_center.w   + P[id_p1].w  ) - (1.0/12.0)*(P[id_m1].w   + P[id_p2].w  );
                    double P_r_P   = (7.0/12.0)*(p_center.P   + P[id_p1].P  ) - (1.0/12.0)*(P[id_m1].P   + P[id_p2].P  );

                    double P_l_rho = (7.0/12.0)*(P[id_m1].rho + p_center.rho) - (1.0/12.0)*(P[id_m2].rho + P[id_p1].rho);
                    double P_l_u   = (7.0/12.0)*(P[id_m1].u   + p_center.u  ) - (1.0/12.0)*(P[id_m2].u   + P[id_p1].u  );
                    double P_l_v   = (7.0/12.0)*(P[id_m1].v   + p_center.v  ) - (1.0/12.0)*(P[id_m2].v   + P[id_p1].v  );
                    double P_l_w   = (7.0/12.0)*(P[id_m1].w   + p_center.w  ) - (1.0/12.0)*(P[id_m2].w   + P[id_p1].w  );
                    double P_l_P   = (7.0/12.0)*(P[id_m1].P   + p_center.P  ) - (1.0/12.0)*(P[id_m2].P   + P[id_p1].P  );

                    auto [rho_L, rho_R] = monetize(P_l_rho, P_r_rho, p_center.rho);
                    auto [u_L, u_R]     = monetize(P_l_u,   P_r_u,   p_center.u);
                    auto [v_L, v_R]     = monetize(P_l_v,   P_r_v,   p_center.v);
                    auto [w_L, w_R]     = monetize(P_l_w,   P_r_w,   p_center.w);
                    auto [P_L, P_R]     = monetize(P_l_P,   P_r_P,   p_center.P);
                    
                    UF[id]  = PrimToCons({rho_L, u_L, v_L, w_L, P_L});
                    UBa[id] = PrimToCons({rho_R, u_R, v_R, w_R, P_R});
                }

            }
        }
    }

}
////////////////////////////////////////////////////////////////////////////
///////////////////////////  Boundary  /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Apply_Boundary(vector<ConsState>& U) {
    // X
    #pragma omp parallel for collapse(2) 
    for(int k = 0; k < Nz_tot ; k++){   
        for(int j = 0; j < Ny_tot ; j++){
            for(int g = 0 ; g < nghost ; g++){
                // outflow
                // U[idx(g,j)]                    = U[idx(nghost,j)];
                // U[idx(Nx_tot -  nghost + g,j)] = U[idx(Nx_tot - nghost - 1, j)];

                // periodic
                U[idx(g, j , k)]                 = U[idx(Nx+g,j,k)];
                U[idx(Nx_tot - nghost+g,j,k)] = U[idx(nghost+g,j,k)];
            }
        }
    }
    
    // Y
    #pragma omp parallel for collapse(2)
    for(int k = 0; k < Nz_tot ; k++){   
        for(int g = 0 ; g < nghost ; g++){
            for(int i = 0; i < Nx_tot ; i++){
                // outflow
                // U[idx(i,g)]                    = U[idx(i, nghost)];
                // U[idx(i, Ny_tot - nghost + g)] = U[idx(i, Ny_tot - nghost - 1)];

                // periodic
                U[idx(i , g , k )]                 = U[idx(i,Ny+g , k)];
                U[idx(i , Ny_tot - nghost+g , k)]  = U[idx(i,nghost+g , k)];
            }
        }
    }


    // Z
    #pragma omp parallel for collapse(2) 
    for(int g = 0 ; g < nghost ; g++){  
        for(int j = 0; j < Ny_tot ; j++){
           for(int i = 0; i < Nx_tot ; i++){   
                // outflow
                // U[idx(g,j)]                    = U[idx(nghost,j)];
                // U[idx(Nx_tot -  nghost + g,j)] = U[idx(Nx_tot - nghost - 1, j)];

                // periodic
                // U[idx(i,j,g)]                 = U[idx(i,j,Nz+g)];
                // U[idx(i,j,Nz_tot - nghost+g)] = U[idx(i,j,nghost+g)];

                int ghost_low = g;
                int fluid_low = 2 * nghost - 1 - g; 
                
                U[idx(i,j,ghost_low)]    =   U[idx(i,j,fluid_low)];
                U[idx(i,j,ghost_low)].mw = - U[idx(i,j,fluid_low)].mw;

                int ghost_up = Nz_tot - nghost + g;
                int fluid_up = Nz_tot - nghost - 1 - g; 
                
                U[idx(i,j,ghost_up)]    =   U[idx(i,j,fluid_up)];
                U[idx(i,j,ghost_up)].mw = - U[idx(i,j,fluid_up)].mw;
            }
        }
    }
}
//===========================================================================
inline ConsState HLLC_Flux(const ConsState& UL_state, const ConsState& UR_state, char direction) {
    PrimState PL = ConsToPrim(UL_state);
    PrimState PR = ConsToPrim(UR_state);

    double aL = sqrt(gam * PL.P / PL.rho);
    double aR = sqrt(gam * PR.P / PR.rho);

    double uL, uR;
    if (direction == 'x') {
        uL = PL.u;
        uR = PR.u;
    } else if (direction == 'y') {
        uL = PL.v;
        uR = PR.v;
    } else { // 'z'
        uL = PL.w;
        uR = PR.w;
    }

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
                UL_star.mw = factor * PL.w; 
            } else if (direction == 'y') {
                UL_star.mu = factor * PL.u;
                UL_star.mv = factor * SM;
                UL_star.mw = factor * PL.w; 
            } else { 
                UL_star.mu = factor * PL.u;
                UL_star.mv = factor * PL.v;
                UL_star.mw = factor * SM;
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
                UR_star.mw = factor * PR.w; 
            } else if (direction == 'y') {
                UR_star.mu = factor * PR.u;
                UR_star.mv = factor * SM;
                UR_star.mw = factor * PR.w; 
            } else { 
                UR_star.mu = factor * PR.u;
                UR_star.mv = factor * PR.v;
                UR_star.mw = factor * SM;
            }
            UR_star.E = factor * (UR_state.E / PR.rho + (SM - uR) * (SM + PR.P / (PR.rho * (SR - uR) + rho_f)));
            return fR + (UR_star - UR_state) * SR;
        }
    }
}
// /////////////////////////////////////////////////////////////////////////////
// /////////////////////////////  PDE Operator  ////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////


void Euler_Riemann_Operator(const vector<ConsState>& U , vector<ConsState>& dUdt,  
    vector<ConsState>& UL,      vector<ConsState>& UR,
    vector<ConsState>& UT,      vector<ConsState>& UB,
    vector<ConsState>& UF,      vector<ConsState>& UBa,
    vector<ConsState>& Flux_x,  vector<ConsState>& Flux_y , vector<ConsState>& Flux_z ){
    
    //===========================================================================
    // 1 : data reconstruction | 0.035 s

    // Data_Reconstruct_PLM(U,UL,UR,UT,UB,UF,UBa);
    Data_Reconstruct_PPM(U,UL,UR,UT,UB,UF,UBa);
    
    //===========================================================================
    // 2 : Flux 
    // #pragma omp parallel for collapse(3)
    // for(int k = nghost; k <= Nz_tot-nghost; k++){   
    //     for (int j = nghost; j <= Ny_tot-nghost; j++) {
    //         for (int i = nghost; i <= Nx_tot-nghost; i++) {
    //             int id = idx(i, j, k);

    //             if (k < Nz_tot-nghost && j < Ny_tot-nghost) {
    //                 Flux_x[id] = HLLC_Flux(UR[idx(i-1, j, k)], UL[id], 'x');
    //             }

    //             if (k < Nz_tot-nghost && i < Nx_tot-nghost) {
    //                 Flux_y[id] = HLLC_Flux(UT[idx(i, j-1, k)], UB[id], 'y');
    //             }

    //             if (j < Ny_tot-nghost && i < Nx_tot-nghost) {
    //                 Flux_z[id] = HLLC_Flux(UF[idx(i, j, k-1)], UBa[id], 'z');
    //             }
    //         }
    //     }
    // }

    // X
    #pragma omp parallel for collapse(3)
    for(int k = 0; k < Nz_tot ; k++){   
        for (int j = nghost; j < Ny_tot-nghost; j++) {
            for (int i = nghost; i <= Nx_tot-nghost; i++) {
                int id_L = idx(i-1, j , k);
                int id_R = idx(i, j , k);
                Flux_x[id_R] = HLLC_Flux(UR[id_L], UL[id_R], 'x');

                // Flux_x[idx(i, j , k)] = HLLC_Flux(UR[idx(i-1, j , k)], UL[idx(i, j , k)], 'x');
            }
        }
    }

    // Y
    #pragma omp parallel for collapse(3)
    for(int k = 0; k < Nz_tot ; k++){   
        for (int j = nghost; j <= Ny_tot-nghost; j++) {
            for (int i = nghost; i < Nx_tot-nghost; i++) {
                int id_B = idx(i, j-1 , k);
                int id_T = idx(i, j , k);
                Flux_y[id_T] = HLLC_Flux(UT[id_B], UB[id_T] , 'y');
                // Flux_y[idx(i, j , k)] = HLLC_Flux(UT[idx(i, j-1 , k)], UB[idx(i, j , k)], 'y');
            }
        }
    }

    // Z
    #pragma omp parallel for collapse(3)
    for(int k = nghost; k <= Nz_tot-nghost ; k++){   
        for (int j = nghost; j < Ny_tot-nghost; j++) {
            for (int i = nghost; i < Nx_tot-nghost; i++) {
                int id_F  = idx(i, j , k-1);
                int id_Ba = idx(i, j , k);
                Flux_z[id_Ba] = HLLC_Flux(UF[id_F], UBa[id_Ba], 'z');
                // Flux_z[idx(i, j , k)] = HLLC_Flux(UF[idx(i, j , k-1)], UBa[idx(i, j , k)], 'z');
            }
        }
    }

    //===========================================================================
    // 3: dU/dt 

    #pragma omp parallel for collapse(3)
    for(int k = nghost; k < Nz_tot-nghost ; k++){   
        for (int j = nghost; j < Ny_tot-nghost; j++) {
            for (int i = nghost; i < Nx_tot-nghost; i++) {

                ConsState dFx_dx = (Flux_x[idx(i+1, j ,k )] - Flux_x[idx(i, j , k)]) * inv_dx;
                ConsState dGy_dy = (Flux_y[idx(i, j+1 ,k )] - Flux_y[idx(i, j , k)]) * inv_dy;
                ConsState dHz_dz = (Flux_z[idx(i, j,k + 1)] - Flux_z[idx(i, j , k)]) * inv_dz;

                dUdt[idx(i, j ,k)] = -(dFx_dx + dGy_dy + dHz_dz );
            }
        }
    }
}


// /////////////////////////////////////////////////////////////////////////////
// /////////////////////////////  RK4 Solver  //////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// RK2
vector<ConsState> k1(Nx_tot * Ny_tot * Nz_tot);
vector<ConsState> k2(Nx_tot * Ny_tot * Nz_tot);

vector<ConsState> UL(Nx_tot * Ny_tot * Nz_tot ) , UR(Nx_tot * Ny_tot * Nz_tot ) ;
vector<ConsState> UT(Nx_tot * Ny_tot * Nz_tot ) , UB(Nx_tot * Ny_tot * Nz_tot ) ;
vector<ConsState> UF(Nx_tot * Ny_tot * Nz_tot ) , UBa(Nx_tot * Ny_tot * Nz_tot ) ;
vector<ConsState> Flux_x(Nx_tot * Ny_tot * Nz_tot) , Flux_y(Nx_tot * Ny_tot * Nz_tot) ,Flux_z(Nx_tot * Ny_tot * Nz_tot) ;

void RK2_Step(const vector<ConsState>& Un, vector<ConsState>& Unext, double dt) {
    
    // Riemman Solver 0.08s
    // double s = omp_get_wtime();
    Euler_Riemann_Operator(Un, k1,UL,UR,UT,UB,UF,UBa,Flux_x,Flux_y,Flux_z);
    // cout << "ERO  " <<omp_get_wtime() - s << endl;

    // s = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot * Nz_tot; i++) {
        Unext[i] = Un[i] + k1[i] * dt;
    }
    // cout << "Un   " <<omp_get_wtime() - s << endl;

    // s = omp_get_wtime();
    Apply_Boundary(Unext);
    // cout << "AB   " <<omp_get_wtime() - s << endl;

    Euler_Riemann_Operator(Unext, k2,UL,UR,UT,UB,UF,UBa,Flux_x,Flux_y,Flux_z);

    // Floor check 0.001s
    #pragma omp parallel for
    for (int i = 0; i < Nx_tot * Ny_tot * Nz_tot; i++) {
        Unext[i] = Un[i] + (k1[i] + k2[i]) * 0.5 * dt;
    }

    Apply_Boundary(Unext);

}

// void RK4_Step(
//     const vector<ConsState>& Un, 
//     vector<ConsState>& Unext,
//     vector<ConsState>& k1, vector<ConsState>& k2, vector<ConsState>& k3, vector<ConsState>& k4,
//     vector<ConsState>& U_temp,
//     double dt) 
// {
//     // Stage 1
//     NSE_Operator(Un, k1);

//     // Stage 2
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k1[i] * (0.5 * dt);
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k2);

//     // Stage 3
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k2[i] * (0.5 * dt);
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k3);

//     // Stage 4
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k3[i] * dt;
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k4);

//     // Final Integration
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) {
//         Unext[i] = Un[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
//     }
// }


///////////////////////////////////////////////////////////////////

int main(){

    vector<double> x(Nx_tot) , y(Ny_tot) ,z(Nz_tot);
    vector<ConsState> u0(Nx_tot * Ny_tot * Nz_tot);

    // setting grid
    Grid_initail(x,y,z);

    // // initial value
    Initial_Value(u0, x, y, z);

    Apply_Boundary(u0);

    vector<ConsState> u_next(Nx_tot * Ny_tot * Nz_tot) ;
    double t_start = omp_get_wtime();

    for (int step=1 ; step <= total_step ; step++){

        double dt = Get_CFL_Dt(u0);

        t += dt;

        RK2_Step(u0,u_next,dt);

        u0.swap(u_next);

        if (step % Estep == 0) {
            double t_end = omp_get_wtime();
            total_calc_time += (t_end - t_start);

            char buffer[50];
            std::snprintf(buffer, sizeof(buffer), "%04d", step / Estep);
            std::string stepStr = buffer;
            
            cout << "==========================================================\n";
            cout << " Simulation Time = " << t << " , dt = " << dt << endl;
            cout << "            Step = " << step << " | Center density rho = " << u0[idx(Nx_tot/2,Ny_tot/2,Nz_tot/2)].rho << "\n";
            cout << "       Real Time = " << total_calc_time << " s | RK2 Spend Time = " << t_end - t_start << " s\n";

            string Fname = "Final.csv";

            ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + stepStr + "_" + Fname);
            outFile << "time,x,y,z,rho,u,v,w,E,P\n";

            for(int k = nghost; k < Nz_tot-nghost; k++) {
                for(int j = nghost; j < Ny_tot-nghost; j++) {
                    for(int i = nghost; i < Nx_tot-nghost; i++) {
                        double id = idx(i,j,k);
                        PrimState P = ConsToPrim(u0[id]);

                        outFile << t            << " , ";
                        outFile << x[i]         << " , " << y[j]  << " , " << z[k]   << " , " << u0[id].rho  << " , ";
                        outFile << P.u          << " , " << P.v   << " , " << P.w    << " , " ;
                        outFile << u0[id].E     << " , " << P.P ;
                        outFile << '\n';
                    }
                }
            }
            outFile.close();
            t_start = omp_get_wtime();
        }
        
    }
    cout << "\n" << prob << endl;
}

