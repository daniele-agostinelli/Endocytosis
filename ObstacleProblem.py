''' This code implements an obstacle problem for the Canham-Helfrich functional, via a Monge parameterization'''

from __future__ import print_function
from fenics import *
from dolfin import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Obstacle(UserExpression):
    """ Obstacle function, which depends on three parameters only: 
    1. the length ratios rbar=r/l
    2. the Rbar=R/l
    3. the indentation delta=2*d*r
    where r= inhibitor radius, R = particle radius, and l = ligand spacing = 1/\sqrt(xi_L)
    """
    def eval(self, value, x):
        if x[0]*x[0] < rbar**2:
            val_pr = sqrt(rbar**2-x[0]*x[0])+(2*d-1)*rbar+(1-cos(1/Rbar))*Rbar # upper part of the protrusion
        else:
            val_pr = -inf
        val_pa = sqrt((Rbar)**2-x[0]*x[0]) - cos(1/Rbar)*Rbar                  # upper part of the particle
        value[0] = max(val_pr,val_pa)
    def value_shape(self):
        return ()
    
def NegativePart(x, eps=1e-4):
    """ Approximation of the Negative Part function: x |--> min(x, 0) """
    return conditional(lt(x, -eps), x+eps/2, conditional(gt(x, 0), 0, -x**2/(2*eps)))

def SolveObstacleProblem():
    """ Function to solve the obstacle problem for values of the model parameters that are set globally """
    global hsol, G, AreaP_xiL, AreaS_xiL, DeltaA_xiL
    
    #------------ PROBLEM GEOMETRY -------------
    # Define (dimensionless) domain
    DomL = 0
    DomR = Rbar*sin(1/Rbar)

    # Create mesh
    mesh =  IntervalMesh(2**7,DomL,DomR)
    
    #------------ FEM FORMULATION -------------
    # Define function space
    P2      = FiniteElement('P', mesh.ufl_cell(), 2)
    element = MixedElement([P2,P2])                   # two unknowns: C=2H (two times mean curvature), h (height function of Monge parametrization)
    V       = FunctionSpace(mesh, element)

    # Boundary conditions
    cl = ZERO                                                    # Constant value for Neumann condition on the left: h'(DomL)=cl
    bc = DirichletBC(V.sub(1), ZERO, 'on_boundary && x[0]>0.01') # Dirichlet boundary condition on the right: h(DomR)=0
    hp = -tan(1/Rbar)                                           
    cr = Constant(hp/sqrt(1+hp**2))                              # Constant value for Neumann condition on the right: h'(DomR)=cr

    # Define domain boundaries and mark them with corresponding indices: 0 for left, and 1 for right
    boundary_markers = MeshFunction("size_t", mesh, dim=mesh.topology().dim()-1, value=0)
    boundary_indices = {'Left': 0,
                        'Right': 1}
    Left  = CompiledSubDomain('(x[0]<0.01) && on_boundary') # 0
    Right = CompiledSubDomain('(x[0]>0.01) && on_boundary') # 1
    Left.mark(boundary_markers, boundary_indices["Left"])
    Right.mark(boundary_markers, boundary_indices["Right"])
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers) # define line integral over the boundary to include the unknowns evaluated at the boundary in the variational expression

    # Define variational problem
    U  = Function(V)            # Function for solution
    Uk = U.copy(deepcopy=True)  # Save zero initial guess function for later use
    H,   h   = split(U)         # Two dimensionless unknowns: H (mean curvature), h (height function of Monge parametrization)
    H_t, h_t = TestFunctions(V) 
    g  = rho*(1+h.dx(0)**2)**(1/2)
    K  = h.dx(0)*(2*H-h.dx(0)/g)/g   # Gaussian curvature for post-processing use
    F  = (h_t.dx(0)/(1+h.dx(0)**2))*((2*H-Ceff)*h.dx(0)**2+(3*H**2+lambsq)*h.dx(0)*g-2*rho*H.dx(0))*dx+ p*NegativePart(h-h_ob)*h_t*rho*dx+\
         H_t*(2*H-h.dx(0)/g)*dx+H_t.dx(0)*h.dx(0)*(1+h.dx(0)**2)**(-1/2)*dx-H_t*cr*ds(1)+H_t*cl*ds(0) # variational expression 
    dU = TrialFunction(V)
    J  = derivative(F,U,dU)
    problem = NonlinearVariationalProblem(F, U, bc, J) # define problem

    # Calculate solution with FEniCS Newton solver
    for i in np.linspace(0,1,100):
        p.t = i  # linear increase of the penalty parameter
        try:
            solver = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = 1e-11
            prm["newton_solver"]["relative_tolerance"] = 1e-14
            prm["newton_solver"]["maximum_iterations"] = 1000
            solver.solve()
        except:
            print('Warning: error encountered in the Newton solver. Trying again with a zero initial guess...')
            U.assign(Uk)            # zero initial guess
            solver = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = 1e-11
            prm["newton_solver"]["relative_tolerance"] = 1e-14
            prm["newton_solver"]["maximum_iterations"] = 1000
            solver.solve()
    
    # calculate terms for post-processing
    # calculate the terms associated with the bending energy of the current configuration of the membrane 
    AreaP_xiL  = 2*pi*assemble(g*dx)            # [-] Dimensionless area of the membrane wrapping the protrusion
    AreaS_xiL  = 2*pi*(1-cos(1/Rbar))*(Rbar**2) # [-] Dimensionless area of the projection of the membrane on the spherical surface 
    DeltaA_xiL = AreaP_xiL-AreaS_xiL            # [-] Difference of dimensionless areas
    integrand = 2*Bm*H**2+2*Bc*(H-Hstar_bar)**2+Bkc*K
    G         = 2*pi*assemble(integrand*g*dx)
    
    # update solution for plotting and saving
    Hsol, hsol = U.split() 
    
    return

def drawSphere(xCenter, yCenter, zCenter, r, angle0, angle1):
    """ Calculate coordinates of a spherical portion with radius r, 
    centered in (xCenter,yCenter,zCenter), and defined by the two angles angle0 and angle1 """
    N = 2**7+1 # number of points for plot
    t = np.linspace(0, 2*np.pi, N)     # theta - parametrizing angle
    p = np.linspace(angle0, angle1, N) # phi   - parametrizing angle
    x = np.outer(np.cos(t), np.cos(p))
    y = np.outer(np.sin(t), np.cos(p))
    z = np.outer(np.ones(np.size(t)), np.sin(p))
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def drawHalfSphere(xCenter, yCenter, zCenter, r, angle0, angle1, ang):
    """ Calculate coordinates of the two halves (back and front, according to an angle ang) of a spherical portion
    with radius r, centered in (xCenter,yCenter,zCenter), and defined by the two angles angle0 and angle1. """
    N  = 2**7+1 # number of points for plot
    t1 = np.linspace(ang, pi+ang, N)      # theta - parametrizing angle
    t2 = np.linspace(pi+ang, 2*pi+ang, N) # theta - parametrizing angle
    p  = np.linspace(angle0, angle1, N)   # phi   - parametrizing angle
    # coordinates of first half
    x1 = np.outer(np.cos(t1), np.cos(p))
    y1 = np.outer(np.sin(t1), np.cos(p))
    z1 = np.outer(np.ones(np.size(t1)), np.sin(p))
    # coordinates of second half
    x2 = np.outer(np.cos(t2), np.cos(p))
    y2 = np.outer(np.sin(t2), np.cos(p))
    z2 = np.outer(np.ones(np.size(t2)), np.sin(p))
    # Shift and scale sphere
    x1 = r*x1 + xCenter
    y1 = r*y1 + yCenter
    z1 = r*z1 + zCenter
    x2 = r*x2 + xCenter
    y2 = r*y2 + yCenter
    z2 = r*z2 + zCenter
    return (x1,y1,z1,x2,y2,z2)

def Plot3D(save=True, show=False, folder='movie', Elev=12, Azim=-59):  
    """ Reconstruct 3D plot from the 1D solution, by exploiting the radial symmetry """
    # Define (dimensionless) domain
    DomL = 0
    DomR = sin(1/Rbar)*Rbar
    
    # 3D coordinates for cell membrane (1 <-> back, 2 <-> front)
    hsol.set_allow_extrapolation(True)
    RHO   = np.linspace(DomL,DomR,101)
    THETA = np.linspace(0,2*pi,101)
    ang   = pi/2+Azim*pi/180
    THETA1 = np.linspace(ang,pi+ang,101)
    THETA2 = np.linspace(pi+ang,2*pi+ang,101)

    X = np.outer(RHO,np.cos(THETA))
    Y = np.outer(RHO,np.sin(THETA))
    Z = np.outer(np.array([hsol(rho) for (rho,theta) in zip(RHO,THETA)]),np.ones(np.size(THETA)))

    X1 = np.outer(RHO,np.cos(THETA1))
    Y1 = np.outer(RHO,np.sin(THETA1))
    Z1 = np.outer(np.array([hsol(rho) for (rho,theta) in zip(RHO,THETA1)]),np.ones(np.size(THETA1)))

    X2 = np.outer(RHO,np.cos(THETA2))
    Y2 = np.outer(RHO,np.sin(THETA2))
    Z2 = np.outer(np.array([hsol(rho) for (rho,theta) in zip(RHO,THETA2)]),np.ones(np.size(THETA2)))

    # 3D coordinates for protrusion
    X_Pr, Y_Pr, Z_Pr = drawSphere(0,0,(2*d-1)*rbar+(1-cos(1/Rbar))*Rbar,rbar,-np.pi/2,np.pi/2) #
    
    # 3D coordinate for particle
    X_Pa1, Y_Pa1, Z_Pa1, X_Pa2, Y_Pa2, Z_Pa2 = drawHalfSphere(0,0,-cos(1/Rbar)*Rbar,Rbar, np.pi/2,np.pi/2+1/Rbar, ang)

    # Plot settings      
    Nstrides_ce = 8 # number of strides for cell membrane 
    Nstrides_pa = 6 # number of strides for particle    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('$\\bar{r}$=%.2e, $\\bar{R}$= %.2e, $d=%.2e$' % (rbar,Rbar,d),fontsize=20)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Plot cell membrane (back)
    ax.plot_surface(X1, Y1, Z1, color='#05bd55', alpha=.1,  linewidth=1,antialiased=False, shade=True,label='Cell')
    ax.plot_wireframe(X1, Y1, Z1, color='#05bd55', alpha=.75,cstride=Nstrides_ce,rstride=Nstrides_ce,  linewidth=1,antialiased=False, label='Cell')
    
    # Plot particle
    ax.plot_surface(X_Pa1, Y_Pa1, Z_Pa1 , color='#5599ff', cstride=Nstrides_pa, rstride=Nstrides_pa, linewidth=0,antialiased=False, alpha=1, shade=True, label='Particle') # back
    ax.plot_surface(X_Pa2, Y_Pa2, Z_Pa2 , color='#5599ff', cstride=Nstrides_pa, rstride=Nstrides_pa, linewidth=0,antialiased=False, alpha=1, shade=True, label='Particle') # front

    # Plot inhibiting protrusion
    ax.plot_surface(X_Pr, Y_Pr, Z_Pr , color='#aa0000', linewidth=0,antialiased=False,alpha=1, shade= True, label='Protrusion')
    
    # Plot cell membrane (front)
    ax.plot_surface(X2, Y2, Z2, color='#05bd55', alpha=.1, linewidth=1,antialiased=False, shade=True,label='Cell')
    ax.plot_wireframe(X2, Y2, Z2, color='#05bd55', alpha=.75,cstride=Nstrides_ce,rstride=Nstrides_ce, linewidth=1,antialiased=False, label='Cell')

    # Other plot settings 
    XYlim = [-1,1]
    Zlim  = [0,2]
    ax.set_xlim3d(XYlim)
    ax.set_ylim3d(XYlim)
    ax.set_zlim3d(Zlim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev = Elev, azim = Azim)

    fig.patch.set_visible(False)
    ax.axis('off')
    
    # Save
    if save:
        fig.savefig('Solution3D.svg')
        plt.close(fig) # close plot
    # Show
    if show:
        plt.show() # hold plot    

def Plot(save=True, show=False, folder='movie'):     
    z_ob = Obstacle()

    # Plot result
    fig = plt.figure() #figsize=(height,width)
    plt.title('$\\bar{r}$=%.2e, $\\bar{R}$= %.2e, $d$=%.2e' % (rbar,Rbar,d),fontsize=20)
    
    # define non-dimensional domain
    DomL = 0
    DomR = sin(1/Rbar)*Rbar
    # Curves plot
    tol = 0.001  # avoid hitting points outside the domain
    points = np.linspace(DomL + tol, DomR - tol, 101)
    h_radial = np.array([hsol(point) for point in points])
    plt.plot(points, h_radial, color='#05bd55', linewidth=2,zorder=4)
    
    x1 = np.linspace(DomL,DomR,101)
    x2 = np.linspace(-rbar,rbar,101)
    virus = np.sqrt((Rbar)**2-x1**2) -cos(1/Rbar)*Rbar

    antibody_up = np.sqrt(rbar**2-x2**2) +(2*d-1)*rbar+(1-cos(1/Rbar))*Rbar
    antibody_dn = -np.sqrt(rbar**2-x2**2)+(2*d-1)*rbar+(1-cos(1/Rbar))*Rbar
    z_obs = np.array([z_ob(point) for point in points])

    plt.xlabel('$\\bar{\\rho}$')
    plt.ylabel('$\\bar{h}$')
    plt.plot(x1, virus,color='#5599ff', linewidth=1.5,zorder=2)
    plt.plot(x2, antibody_up,color='#aa0000', linewidth=1.5,zorder=3)
    plt.plot(x2, antibody_dn,color='#aa0000', linewidth=1.5,zorder=3)
    plt.plot(-points, h_radial, color='#05bd55', linewidth=2,zorder=4)
    plt.plot(-x1, virus,color='#5599ff', linewidth=1.5,zorder=2)
    plt.plot(points, z_obs, 'y', linewidth=1) 
    plt.plot(-points, z_obs, 'y', linewidth=1) 

    plt.legend(['Cell','Particle','Inhibitor'], loc='upper right')
    plt.xlim([-1,1])
    plt.ylim([0,2])
    axes = plt.gca()
    axes.set_aspect(1)

    # Save
    if save:
        fig.savefig('Solution2d.svg')
        plt.close(fig) # close plot
    # Show
    if show:
        plt.show() # hold plot        
        
        
############################################## EXAMPLE ##########################################        
# Define Universal Constants
PEN    = 1e8         # Penalty coefficient
ZERO   = Constant(0) # Dolfin constant

#------------ INPUT PARAMETERS -------------
# Geometrical parameters
xi_L   = 5000*1e-6        # [1/nanometre^2] (from Gao) ligand surface density on viruses - used for non-dimensionalizing quantities
d      = 1                # [-] dimensioless factor of spherical protrusion
rbar   = 0.2              # [-] dimensionless shape ratio of the obstacle
Rbar   = 4                # [-] dimensionless particle radius

# Mechanical Parameter
sigma_bar = 1     # [-] dimensionless membrane tension
Bm        = 20    # [-] Bending modulus of the membrane
Clathrin  = False # True to activate clathrin coat
if Clathrin:
    Bc    = 300                     # [-] Bending modulus of the clathrin coating
    Hstar_bar = -(1/50)/sqrt(xi_L)  # [-] spontaneous curvature of the clathrin coating # Must be negative/positive according to the choice of the normal unit vector (Negative for Monge Parametrization)
    Bkc   = 0                       # [-] Saddle-Splay Gaussian modulus of the clathrin coating
else:
    Bc, Hstar_bar, Bkc = 0, 0, 0

lambsq = Constant((sigma_bar+2*Bc*Hstar_bar**2)/(Bm+Bc))  # [-] square of the dimensionless ratio between membrane tension + spontaneous curvature and bending rigidities
Ceff   = Constant(2*Bc*Hstar_bar/(Bm+Bc))                 # [-] dimensionless effective spontaneous curvature

#---------- FEM SOLUTION -------------- 
# Define global variational functions
p    = Expression('p0*(1-t)+p1*t',t=1,p0=1e3,p1=PEN,degree=1) # Penalty parameter
rho  = Expression('x[0]',degree=1) # [-] Define the dimensionless radial coordinate
h_ob = Obstacle()                  # [-] Define the dimensionless obstacle function. This is updated when changing the global values of the obstacle d, Rbar and rbar

# Solve problem and time computational solver
start = time.time()
SolveObstacleProblem()
end = time.time()
print('Computation time: %g sec' %(end-start))

# 3D and 2D plots
Plot3D(save=False, show=True)
Plot(save=True, show=True)
