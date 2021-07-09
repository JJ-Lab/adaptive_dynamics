#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### 12/05/2021

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc
import random
import os
import sys
import psutil

gammalist = [0.1] # gamma list
n_alphas= 10  # alpha tries

r02max = 1.0
b0imax = 0.01
u0max=0.99

#############
nesp = 2 #number of species
nt = 1000 #total resources
ee = 1/nt #0.001. independent of nt
mu = 1/(ee**2) # mu*ee^2=1
iters = 10000000 # max_iterations

scale = 5 
dt = 10**(-scale) 

r0_str= str(r02max).replace('0.', 'p').replace('1.0','1p')
b0_str= str(b0imax).replace('0.', 'p') 
u0_str= str(u0max).replace('0.', '')

###demographic parameters
a = [0.001 for i in range(nesp)] 
c = [0.001 for i in range(nesp)]

l = [1 for i in range(3)] #(np.random.uniform(1,2, size = nesp).tolist() Parameter in adaptive dynamics equations
##############################################################################################
t0 = 10000 # t  for stationarity
prec = 10000 # precision  

t1 = iters 

ulist = np.zeros((t1,2), dtype=float)
wlist = np.zeros((t1,1), dtype=float)
alphalist = np.zeros((t1,2), dtype=float)
rlist = np.zeros((t1,2), dtype=float)
blist = np.zeros((t1,2), dtype=float)
xoyolist = np.zeros((t1,2), dtype=float)
xoyoralist = np.zeros((t1,2), dtype=float)

##### parameters for plotting
n_steps_plot = 1
n_plots = int(iters/n_steps_plot)

ulist2 = np.zeros((n_plots,2), dtype=float)
wlist2 = np.zeros((n_plots,1), dtype=float)
alphalist2 = np.zeros((n_plots,2), dtype=float)
rlist2 = np.zeros((n_plots,2), dtype=float)
blist2 = np.zeros((n_plots,2), dtype=float)
xoyolist2 = np.zeros((n_plots,2), dtype=float)
xoyoralist2 = np.zeros((n_plots,2), dtype=float)

def ini():
    global ini_ok,ulist,wlist,rlist,blist,xoyolist,alphalist,r0,b0,r0_str,b0_str,r,b,u,w,a,c,l,nt,ee,mu,alpha,gamma,xoyoralist
    global r0_rand, b0_rand, u0_rand, w0_rand, alpha0_rand, n_alphas, i_list, i_overflow, i_plot, i_rand, n_rand
    global u_ini, w_ini, alpha_ini, r_ini, b_ini, xoyo_ini

    i_list = 0
    i_overflow = 0
    i_plot = 0
    found = False    
    
    ### random initial parameters: r0,b0,u0,w0,alpha0
    r0 = [1, float(np.random.uniform(0,r02max, size = 1))]
    b0 = np.random.uniform(0,b0imax, size = 2).tolist() # [0.05*i for i in ro] #
    
    u0 = np.random.uniform(0,u0max, size = 2) #para n=500, usamos (0,0.1), para n=10, (0,0.005), para n=50, (0,0.02)
    w0 = np.random.uniform(0,min(u0[0],u0[1]), size = 1)
    alpha0 = np.random.uniform(0,gamma, size = 2) #los alphas se inicializan en cualquier valor entre [0,gamma]
    
    ### if you want a particular initial condition uncomment
    #[r0,b0,u0,w0,alpha0] = [[1.0, 1.0], [0.0071346273983588105, 0.0030891408219675086], [0.38958202405323583, 0.9635200158129497], [0.05325455889893622], [0.1084824534937299, 0.08866162195952548]]
    
    (r,b,pops,found) = stat_popul(u0,w0,alpha0)
                    
    if found and pops[0]>1 and pops[1]>1:
        xoyo = pops
        update(u0,w0,alpha0,r,b,xoyo) 
        if xoyo[0]*b[1]*c[1] > -a[1] and  xoyo[1]*b[0]*c[0] > -a[0] :
            ini_ok = True
            print('ini Ok: ')
            u_ini = u0
            w_ini = w0
            alpha_ini = alpha0 
            r_ini = r
            b_ini = b
            xoyo_ini = xoyo 
            print('u_ini,w_ini,alpha_ini, r_ini, b_ini, xoyo_ini=', u_ini,w_ini,alpha_ini, r_ini, b_ini, xoyo_ini)
        else:
            print('ini_ok=False; false solution: xoyo[0]*bf[1]*c[1]=',xoyo[0]*b[1]*c[1],'<-a[1]=', -a[1],'or xoyo[1]*bf[0]*c[0]=',xoyo[1]*b[0]*c[0],'<-a[0]=',-a[0], '\n')
            ini_ok = False
    else:
        ini_ok = False
    return ini_ok

##################

def update(u,w,alpha,r,b,xoyo):
    global ulist,wlist,alphalist,rlist,blist,xoyolist,xoyoralist, i_list, i_overflow, t1, iters

    ulist[i_list] = u
    wlist[i_list] = w
    alphalist[i_list] = alpha
    rlist[i_list] = r
    blist[i_list] = b
    xoyolist[i_list] = xoyo
    xoyoralist[i_list] = np.array([max(0,r[0])/a[0], max(0,r[1])/a[1]])
        
    i_list += 1

def popul_pos(u,w,alpha):
    global r,b,equs,solus,jacs,lambdas,index,xoyo,totlambdas,totpops,gamma,r0,b0
    global sol,solus1, solus2

    r = [r0[0]*((1 - gamma - alpha[0]*alpha[1])*u[0] - gamma*alpha[0]*u[1])/(1 - alpha[0]*alpha[1]),
         r0[1]*((1 - gamma - alpha[0]*alpha[1])*u[1] - gamma*alpha[1]*u[0])/(1 - alpha[0]*alpha[1])]
    b = [b0[0]*(((alpha[0]*alpha[1])*u[0] + alpha[0]*u[1])/(1 - alpha[0]*alpha[1]) - w[0]),
         b0[1]*(((alpha[0]*alpha[1])*u[1] + alpha[1]*u[0])/(1 - alpha[0]*alpha[1]) - w[0])]
    #print('r:',r, ', b:',b)
    
    discrim = 4*b[1]*(b[0]*c[0] + a[0]*c[1])*(a[1]*r[0] + b[0]*r[1]) + (a[0]*a[1]-b[1]*(b[0]+c[1]*r[0])+b[0]*c[0]*r[1])**2
    #discrim2= 4*b[0]*(b[1]*c[1] + a[1]*c[0])*(a[0]*r[1] + b[1]*r[0]) + (a[0]*a[1]-b[1]*(b[0]-c[1]*r[0])-b[0]*c[0]*r[1])**2

    if b[0] == 0 or b[1] == 0:
        solus=np.array([0, 0])
    elif discrim < 0 :
        print('complex solution, discriminants',discrim, '\n')
        return (r,b,np.array([]))

    else: 
        solus1 = np.array([(-a[0]*a[1] + b[0]*b[1] + b[1]*c[1]*r[0] - b[0]*c[0]*r[1] + pow(discrim,0.5))/ 
                           (2*b[1]*(b[0]*c[0] + a[0]*c[1])), #N1+
                            (-a[0]*a[1] + b[0]*b[1] + b[1]*c[1]*r[0] - b[0]*c[0]*r[1] - pow(discrim,0.5)) /
                           (2*b[1]*(b[0]*c[0] + a[0]*c[1])) ]) #N1-
                            
        #solus2 = np.array([(-a[0]*a[1] + b[0]*b[1] - b[1]*c[1]*r[0] + b[0]*c[0]*r[1] + pow(discrim,0.5))/ 
        #                   (2*b[0]*(b[1]*c[1] + a[1]*c[0])), #N2+
        #                    (-a[0]*a[1] + b[0]*b[1] - b[1]*c[1]*r[0] + b[0]*c[0]*r[1] - pow(discrim,0.5))/ 
        #                   (2*b[0]*(b[1]*c[1] + a[1]*c[0])) ]) 
    #print('discr:',discrim, ',discr2:',discrim2,' numer:', r[1]+b[1]*solus1, ', denom:', a[1]+c[1]*b[1]*solus1 )

    #### best solution
    solus2_calc = (r[1]+b[1]*solus1)/(a[1]+c[1]*b[1]*solus1)

    #print('solus1:',solus1)
    #print('solus2:',solus2)
    #print('solus2_calc:', solus2_calc)
    sol = []
    for i in range(2):
        if solus1[i]>0 and solus2_calc[i]>0:
            sol.append([solus1[i], solus2_calc[i]])
    #print('sol:', sol)
    solus = np.array(sol)
    #print('solus:',solus)
    
    return (r,b,solus)
############################################


def stat_popul(u,w,alpha):
    global a,c

    pop=np.array([0, 0])
    found = False

    r,b,solus = popul_pos(u,w,alpha)
    #print('solus:',solus, ', shape:',solus.shape, 'type:',type(solus))


    if solus.shape[0]<1:
        #print('Non positive solutions! :', solus)
        found = False
        return (r,b,pop,found)   
      
    jacs=[]
    for k in range(solus.shape[0]):
        jacs.append([[-solus[k][0]*(a[0] + c[0]*b[0]*solus[k][1]),solus[k][0]*b[0]*(1 - c[0]*solus[k][0])],             [solus[k][1]*b[1]*(1 - c[1]*solus[k][1]),-solus[k][1]*(a[1] + c[1]*b[1]*solus[k][0])]]) 
   # print('n. jacs:',k+1)
    lambdas=[]
    
    for jj in jacs:
        #print('jj',jj)
        if not np.all(np.isfinite(np.ndarray.flatten(np.array(jj)))):
            print('Jacs infinite!!: ',jj)
            jacs.remove(jj)
        if np.any(np.isnan(np.ndarray.flatten(np.array(jj)))):
            print('Jacs is NaN!!:',jj)
            jacs.remove(jj)
            #del(jj)

    #lambdas = [np.linalg.eigvals(i) for i in np.array(jacs).astype(np.float64)] 
    
    lambdas = [np.linalg.eigvals(i) for i in np.array(jacs)]           
    #print('lambdas:', lambdas)
    if not (np.all(np.isfinite(lambdas)) ):
        print('lambda non finite:', lambdas)
        return (r, b, np.array([0, 0]), False )
    i_lambd_neg = []
    for i in range(solus.shape[0]):
        if np.all(np.sign(lambdas[i]) < 0):
            i_lambd_neg.append(i) 
            if len(i_lambd_neg)>1: print('i_lamb_neg:',i_lambd_neg, 'lambdas:', lambdas)

    lambd_min=0.
    i_lmin = 0
    for i in range(solus.shape[0]):
        if i in i_lambd_neg:
            min_l = np.min(lambdas[i].real)
            if min_l < lambd_min:
                lambd_min=min_l
                i_lmin=i
                #print('lamb_min:',lambd_min,' en i_lmin:',i_lmin)
    pop = solus[i_lmin]
    #print('the most stable population:',pop)
    found = True
    return (r, b, pop, found)
#########################  

def f_dudt(uf,wf,pops):
    global l,mu,ee,nt,f1u,f2u,f1w,f2w,f1alpha,f2alpha, gamma
#    uf,wf,pops=U
    dudt = [(l[0]*mu*ee*pops[0]/(2*nt))*((f1u-f1w)*uf[1] + abs(f1u-f1w)*(uf[1] - 2*wf[0]) +        f1u*(ee*nt - uf[1]) + abs(f1u)*(ee*nt - 2*uf[0] + 2*wf[0] - uf[1])),
        (l[1]*mu*ee*pops[1]/(2*nt))*((f2u-f2w)*uf[0] + abs(f2u-f2w)*(uf[0] - 2*wf[0]) +\
        f2u*(ee*nt - uf[0]) + abs(f2u)*(ee*nt - 2*uf[1] + 2*wf[0] - uf[0]))]
    return dudt

def f_dwdt (uf,wf,pops):
    global l,mu,ee,nt,f1u,f2u,f1w,f2w,f1alpha,f2alpha, gamma

    dwdt = (l[2]*mu*ee/(2*nt))*(pops[0]*((f1u-f1w)*uf[1] + abs(f1u-f1w)*(uf[1] - 2*wf[0])) +
                                    pops[1]*((f2u-f2w)*uf[0] + abs(f2u-f2w)*(uf[0] - 2*wf[0])))
    return dwdt

def f_dalphadt(alphaf,pops):
    global l,mu,ee,nt,f1u,f2u,f1w,f2w,f1alpha,f2alpha, gamma

    dalphadt = [(l[0]*mu*(ee*ee)*pops[0]/2)*(f1alpha + abs(f1alpha)*(1 - 2*alphaf[0]/gamma)),
    (l[1]*mu*(ee*ee)*pops[1]/2)*(f2alpha + abs(f2alpha)*(1 - 2*alphaf[1]/gamma))]
    return dalphadt

def f_uwalpha(alpha,pops):
    global r0,b0,gamma
    f1u = r0[0]*((1 - gamma - alpha[0]*alpha[1]) + b0[0]*alpha[0]*alpha[1]*pops[1]*(1 - c[0]*pops[0]))/(1 - alpha[0]*alpha[1])
    f1w = b0[0]*pops[1]*(1 - c[0]*pops[0])
    f1alpha = (b0[0]*pops[1]*(1 - c[0]*pops[0]) - gamma*r0[0])/ (1 - alpha[0]*alpha[1])
    f2u = r0[1]*((1 - gamma - alpha[0]*alpha[1]) + b0[1]*alpha[0]*alpha[1]*pops[0]*(1 - c[1]*pops[1]))/(1 - alpha[0]*alpha[1])
    f2w = b0[1]*pops[0]*(1 - c[1]*pops[1])
    f2alpha = (b0[1]*pops[0]*(1 - c[1]*pops[1]) - gamma*r0[1])/ (1 - alpha[0]*alpha[1])
    
    return f1u,f1w,f1alpha,f2u,f2w,f2alpha
#################################################
###############################

def evolution():
    global dt,iters,f1u,f1w,f1alpha,f2u,f2w,f2alpha,dudt,dwdt,dalphadt,fin
    global u,w,alpha,i_alpha,uf,wf,alphaf,dt,gamma, i_list, i_overflow, i_plot
    global ulist,wlist,rlist,blist,xoyolist,alphalist
    global ulist_plot,wlist_plot,rlist_plot,blist_plot,xoyolist_plot,alphalist_plot,xoyoralist_plot 
    fin = False
    for it in range(iters):        
        #if it % 10000 == 0: print('it:', it)
        u0 = ulist[i_list-1]  ## u0 = ulist[i_list-i_overflow-1]
        w0 = wlist[i_list-1]  ## w0 = wlist[i_list-i_overflow-1]
        alpha0 = alphalist[i_list-1]  ## alpha0 = alphalist[i_list-i_overflow-1]
        pops0 = xoyolist[i_list-1]    ## pops0 = xoyolist[i_list-i_overflow-1]
        #####################################################################################
        ### RK4:  y_f = y_i + h*(k1 + 2k2 + 2k3 + k4)/6
        ### k1 = f(x_i,y_i)
        ### k2 = f(x_i+0.5h, y_i + 0.5h*k1)
        ### k3 = f(x_i+0.5h, y_i + 0.5h*k2)
        ### k4 = f(x_i+h, y_i + h*k3)
        ###
        
        
        f1u,f1w,f1alpha,f2u,f2w,f2alpha = f_uwalpha(alpha0,pops0) #se utilizan en las funciones f_dudt, f_dwdt y f_dalphadt 
        # k1 = f(t,y)
        ku1=f_dudt(u0,w0,pops0)
        kw1=f_dwdt(u0,w0,pops0)
        ka1=f_dalphadt(alpha0,pops0)       
       
        #k2 = f(t+0.5dt,y+0.5*k1*dt)
        u2 = [u0[i]+0.5*ku1[i]*dt for i in range(2)]
        w2 = w0+0.5*kw1*dt
        a2 = [alpha0[i]+0.5*ka1[i]*dt for i in range(2)]
        #print('u2,w2,a2: ',u2,w2,a2)

        r2,b2,pops2,found2 = stat_popul(u2,w2,a2)
        if not found2:
            print('Stationary population pops2 Not found')
            f_out.write('Stationary population pops2 not found\n')
            fin = False
            break            
        f1u,f1w,f1alpha,f2u,f2w,f2alpha = f_uwalpha(a2,pops2)
        ku2=f_dudt(u2, w2, pops2)
        kw2=f_dwdt(u2, w2, pops2)
        ka2=f_dalphadt(a2, pops2)        
        
        #k3 = f(t+0.5dt,y+0.5*k2*dt)
        u3 = [u0[i]+0.5*ku2[i]*dt for i in range(2)]
        w3 = w0+0.5*kw2*dt
        a3 = [alpha0[i]+0.5*ka2[i]*dt for i in range(2)]
        #print('u3,w3,a3: ',u3,w3,a3)

        r3,b3,pops3,found3 = stat_popul(u3, w3, a3)
        if not found3:
            print('Stationary population pops3 Not found')
            f_out.write('Stationary population pops3 not found\n')
            fin = False
            break
        f1u,f1w,f1alpha,f2u,f2w,f2alpha = f_uwalpha(a3, pops3)
        ku3=f_dudt(u3, w3, pops3)
        kw3=f_dwdt(u3, w3, pops3)
        ka3=f_dalphadt(a3, pops3)        
                            
        #k4 = f(t+dt,y+k3*dt) 
        u4 = [u0[i]+ku3[i]*dt for i in range(2)]
        w4 = w0+kw3*dt
        a4 = [alpha0[i]+ka3[i]*dt for i in range(2)]
        #print('u4,w4,a4: ',u4,w4,a4)      

        r4,b4,pops4,found4 = stat_popul(u4,w4,a4)        
        if not found4:
            print('Stationary population pops4 Not found')
            f_out.write('Stationary population pops4 not found\n')
            fin = False
            break
        f1u,f1w,f1alpha,f2u,f2w,f2alpha = f_uwalpha([alpha0[i]+ka3[i]*dt for i in range(2)], pops4)
        ku4=f_dudt(u4, w4, pops4)
        kw4=f_dwdt(u4, w4, pops4)
        ka4=f_dalphadt([alpha0[i]+ka3[i]*dt for i in range(2)], pops4)        
            
        # new values of u,v,alpha
        uf = [u0[i] + dt*(ku1[i] + 2.0*ku2[i] + 2.0*ku3[i] + ku4[i])/6.0 for i in range(nesp)]
        wf = w0 + dt*(kw1 + 2.0*kw2 + 2.0*kw3 + kw4)/6.0
        alphaf = [alpha0[i] + dt*(ka1[i] + 2.0*ka2[i] + 2.0*ka3[i] + ka4[i])/6.0 for i in range(nesp)]
        # new satationary population with the new values of u,v,alpha

        (rf,bf,popsf, foundf) = stat_popul(uf,wf,alphaf)
        if not foundf:
            print('Stationary population popsf Not found')
            f_out.write('Stationary population popsf not found\n')
            fin = False
            break
        xoyo=popsf

        if it%(iters/1000) == 0: print('it: ',it, '; after RK, popsf:',popsf, ',uf:',uf,', wf:',wf, ', alphaf:', alphaf)

        if xoyo[0]*bf[1]*c[1] < -a[1] or  xoyo[1]*bf[0]*c[0] < -a[0] :
            f_out.write('false solution' + '\n')
            print('false solution: xoyo[0]*bf[1]*c[1]=',xoyo[0]*bf[1]*c[1], ' <-a[1]=', -a[1], 'or xoyo[1]*bf[0]*c[0]=', xoyo[1]*bf[0]*c[0],'<-a[0]=', -a[0], '\n')
            fin = False
            break

        if np.any([np.imag(rf[i]) for i in range(nesp)]) or np.any([np.imag(bf[i]) for i in range(nesp)]):
            f_out.write('r/b-complex' + '\n')
            print('r/b-complex: r_im=', np.imag(rf),', b_im=',np.imag(bf),'\n')
            fin =  False
            break

        if np.any([np.real(xoyo[i]) > 2*np.real(pops0[i]) for i in range(nesp)]):
            f_out.write('big step in population in it:'+str(it) + 'pops0:' + str(pops0) + ', popsf:' + str(xoyo)+ '\n')
            print('big step in population in it:'+str(it) + 'pops0:' + str(pops0) + ', popsf:' + str(xoyo)+ '\n')

            fin = False
            break

        if np.any([np.real(xoyo[i]) < 1 for i in range(nesp)]):
            f_out.write('extintion with xoyo<1' + '\n')
            print('extintion with xoyo=',xoyo,'<1' + '\n')
            #plotnstats(i,aaa)
            fin = False
            break

        if  it == iters-1:
            f_out.write('NESS: End simulation without stationary solution (iter:'+ str(i_sol) + ')\n')
            #print('NESS: End simulation without stationary solution (iter:'+ str(i_sol) + ')\n')
            fin = True
            update(uf,wf,alphaf,rf,bf,xoyo) ##update(np.array(uf),np.array(wf),np.array(alphaf),np.array(rf),np.array(bf),np.array(xoyo))
            plotnstats(gamma)
            break

        t00 = i_list - t0
        if  it>t0 and np.all([(np.real(uf[j])*prec//1) == (np.real(ulist[t00])[j])*prec//1 for j in range(nesp)]) and                 np.real(wf)*prec//1 == np.real(wlist[t00])*prec//1 and                 np.all([(np.real(alphaf[j])*prec//1) == (np.real(alphalist[t00])[j])*prec//1 for j in range(nesp)]):       
            print('reach ESS with values: ', np.round([ulist[t00][j] for j in range(2)],5),np.round([wlist[t00][0]],5),                  np.round([alphalist[t00][j] for j in range(2)],5) )
            f_out.write('ESS (iter:' + str(it) + ')\n')
            fin = True
            update(uf,wf,alphaf,rf,bf,xoyo)
            plotnstats(gamma)
            break

        update(uf,wf,alphaf,rf,bf,xoyo)  ##update(np.array(uf),np.array(wf),np.array(alphaf),np.array(rf),np.array(bf),np.array(xoyo))
                       
                       
######################################################################################################


def plotnstats(ggg):

    global ulist2,wlist2,rlist2,blist2,xoyolist2,alphalist2, i_list,r0,b0, r_ini,b_ini,u_ini,w_ini,alpha_ini,xoyo_ini
    global ulist_plot,wlist_plot,rlist_plot,blist_plot,xoyolist_plot,alphalist_plot,xoyoralist_plot     
    global dt,i_sol, n_steps_plot
    
    xt = [i*dt for i in range(i_list-1)] ##xt=[i*n_steps_plot*dt for i in range(rlist_plot.shape[0])]
    xtm1 = [i*dt for i in range(i_list-2)]  ##xtm1=[i*n_steps_plot*dt for i in range(rlist_plot.shape[0]-1)]
    
    alph1= np.round(np.real(alphalist[0][0]),3)
    alph2= np.round(np.real(alphalist[0][1]),3)
    alp1 = str(alph1).replace('0.', 'p')
    alp2 = str(alph2).replace('0.', 'p')
    gamm = str(ggg).replace('0.', '')
    #print(alph1,alph2,alp1,alp2)
    name = 'N1000_r'+ r0_str +'_b' + b0_str + '_u' + u0_str + '_gam' + str(gamm)
    name = name + '_i'+ str(i_sol) + 'alph1_' + alp1 + 'alph2_' + alp2 + 'func'
    print('graf_name: ',name)

    fig, ax = plt.subplots()


# using the twinx() for creating another
# axes object for secondry y-Axis
    ax2 = ax.twinx()

#ax.plot(x, y1, color = 'g')
    ax.plot(xt,[x[0] for x in rlist[:i_list-1]],'c--')
    ax.plot(xt,[x[1] for x in rlist[:i_list-1]],'b-.')
#ax2.plot(x, y2, color = 'b')
    ax2.plot(xt,[x[0]*100 for x in blist[:i_list-1]],'m-') # estoy multiplicando b12 y b21 por 100 para re-escalarlo
    ax2.plot(xt,[x[1]*100 for x in blist[:i_list-1]],'y:') # y que aparezcan en el mismo rango de r1 y r2
    plt.legend(['r1','r2','b12(x10^-2)','b21(x10^-2)'], loc='best')
    
# giving labels to the axises
    ax.set_xlabel('time', color = 'r')
    ax.set_ylabel('r', color = 'g')
# secondary y-axis label
    ax2.set_ylabel('b', color = 'b')   
    ax1_ylims = ax.axes.get_ylim()           # Find y-axis limits set by the plotter
    ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  # Calculate ratio of lowest limit to highest limit

    ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
    ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit


# If the plot limits ratio of plot 1 is smaller than plot 2, the first data set has
# a wider range range than the second data set. Calculate a new low limit for the
# second data set to obtain a similar ratio to the first data set.
# Else, do it the other way around

    if ax1_yratio < ax2_yratio: 
        ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
    else:
        ax.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)
    #ax.axhline(0, color='k')
    
    plt.axhline(0, linewidth=0.75, color='black')
        
    ax.legend(['r1','r2'], loc='best')
    ax2.legend(['b1','b2'], loc='best')

#    plt.axis('tight')
    plt.title('Ev. Population param.: r0[' + str(round(r0[0],2))+','+str(round(r0[1],4)) +'], b0[' + str(round(b0[0],4))+','+str(round(b0[1],4)) +']')

# defining display layout 
    plt.tight_layout()
    
    filename = name + '_param'
    plt.savefig(path + filename + '.png')


    plt.clf()
    plt.plot(xt,[x[0]/ee for x in ulist[:i_list-1]],'b-')
    plt.plot(xt,[x[1]/ee for x in ulist[:i_list-1]],'r-')
    plt.plot(xt,[x[0]/ee for x in wlist[:i_list-1]],'g-')
                       
    plt.axis('tight')
    plt.title('Ev. param.: u0[' + str(round(u_ini[0],4))+','+str(round(u_ini[1],4))+ '], w0[' + str(round(w_ini[0],4))+']')
    filename = name  + '_owntraits'
    plt.savefig(path + filename + '.png')
                                                                                                
    plt.clf()
    plt.plot(xt,[x[0] for x in alphalist[:i_list-1]],'c--')
    plt.plot(xt,[x[1] for x in alphalist[:i_list-1]],'m--')
    #plt.plot(range(len(wlist)),[x[0]/ee for x in wlist],'g-')
    plt.legend(['a1','a2'], loc='best')
    plt.axhline(0, linewidth=0.75, color='black')
    plt.axhline(ggg, linewidth=1, color='green')
    plt.axis('tight')
    plt.title('Ev. cross.: alpha0[' + str(round(alpha_ini[0],4))+','+str(round(alpha_ini[1],4))+ ']')                                    
    filename = name  + '_crosstraits'
    plt.savefig(path + filename + '.png')
    plt.clf()
    
    plt.plot(xt,[x[0] for x in xoyolist[:i_list-1]],'g.')
    plt.plot(xt,[x[1] for x in xoyolist[:i_list-1]],'r.')
    plt.plot(xt,[x[0] for x in xoyoralist[:i_list-1]],'g--')
    plt.plot(xt,[x[1] for x in xoyoralist[:i_list-1]],'r--')

    plt.axis('tight')
    plt.title('Ev. popul.: x0y0['+str(int(xoyo_ini[0]))+ ','+str(int(xoyo_ini[1]))+']')                                        
    filename = name  + '_popul'
    plt.savefig(path + filename + '.png')
    plt.clf()
    
    M_evo=np.concatenate((ulist[:i_list-1],wlist[:i_list-1],alphalist[:i_list-1],rlist[:i_list-1],blist[:i_list-1],xoyolist[:i_list-1]),axis=1)
    
    filename2 = name + '_data.txt'
    f_out2 = open(filename2,'w')
    np.savetxt(f_out2, M_evo, fmt='%1.8f', delimiter=',')
    f_out2.close()

######################################################################################################
######################################################################################################
######################################################################################################

i_rand = 0
for j in range(len(gammalist)):
    gamma =gammalist[j]
    gamm = str(gamma).replace('0.', '')
    print('gamma: '+ gamm)

    path_name = 'graph_gamm'+str(gamm)
    if path_name not in os.listdir('.'):
        os.mkdir(path_name)
    path = path_name + '/'

    i_sol=0 #succesful solutions counter  
    while i_sol < n_alphas:        
        ini()
        #i_rand = len(r0_rand_list)-1

        if ini_ok:
            #process = psutil.Process(os.getpid())
            #print('memory before evolution:',(process.memory_info().rss)/1000000)  # in bytes 

            #alph1= np.round(np.real(alphalist[0][0]),3)
            #alph2= np.round(np.real(alphalist[0][1]),3)
            #alp1 = str(alph1).replace('0.', 'p')
            #alp2 = str(alph2).replace('0.', 'p')

            name = 'N1000_r'+ r0_str +'_b' + b0_str + '_u' + u0_str + '_gam' + gamm  + '_nalph'+ str(n_alphas) +'_it' + str(iters) + '_func.txt'
            f_out = open(name,'a')
            #print('ini_ok; i_rand:',i_rand-1, ', f_out name:',name)
            #f_out.write(str(i) + '\t[ro,bo,u,w,alpha]:')
            #param_out_0 = [ro[j] for j in range(2)] + [bo[j] for j in range(2)] + [ulist[0][j] for j in range(2)] + wlist[0].tolist() + [alphalist[0][j] for j in range(2)]  + [xoyolist[0][j] for j in range(2)]  
            param_out_r0b0 = [r0[j] for j in range(2)] + [b0[j] for j in range(2)] 
            param_out_uwalpha0 = [ulist[0][j] for j in range(2)] + wlist[0].tolist() + [alphalist[0][j] for j in range(2)]
            param_out_rb_ini = [rlist[0][j] for j in range(2)] + [blist[0][j] for j in range(2)] 
            param_out_x0y0 = [xoyolist[0][j] for j in range(2)]  
            param_out_0 = str(i_rand) +'\t' + str(i_sol) +'\t' + str(param_out_r0b0)+ str(param_out_uwalpha0) + str(param_out_rb_ini) + str(param_out_x0y0)+'\n'
            f_out.write(param_out_0)
            #print('param_out_0: ', param_out_0)
            #f_out.write(str((ro,bo,u,w,alpha)) )            
            #f_out.write('\t'+ str(dt) + '\n') # write init random parameters
            
            evolution()
            #process = psutil.Process(os.getpid())
            #print('memory after evolution:',(process.memory_info().rss)/1000000)  # in bytes 
        
            if  fin == True:       
                    #f_out.write('r/a,x0,xf:'+str(np.real([xoyoralist[-1],xoyolist[0],xoyolist[-1]])).replace('\n', ',') + '\n') # se escriben las poblaciones r/a y la estacionaria final
                    #u_out_f = np.round([ulist[-1][j] for j in range(2)],5).tolist()+ np.round(wlist[-1],5).tolist() + np.round([alphalist[-1][j] for j in range(2)],5).tolist()

                u_out_0 = [u_ini[j] for j in range(2)] + w_ini + [alpha_ini[j] for j in range(2)]
                f_out.write(str(u_out_0) + '\t')
                u_out_f = [ulist[i_list-1][j] for j in range(2)] + wlist[i_list-1].tolist() + [alphalist[i_list-1][j] for j in range(2)]
                f_out.write(str(u_out_f) + '\n')
                r_b_out_0 = [r_ini[j] for j in range(2)] + [b_ini[j] for j in range(2)]
                f_out.write(str(r_b_out_0) + '\t')
                r_b_out_f = [rlist[i_list-1][j] for j in range(2)] + [blist[i_list-1][j] for j in range(2)]
                f_out.write(str(r_b_out_f) + '\n')
                xoyo_out_0 = np.round([xoyo_ini[j] for j in range(2)],5).tolist() 
                f_out.write(str(xoyo_out_0) + '\t')
                xoyo_out_f = np.round([xoyolist[i_list-1][j] for j in range(2)],5).tolist() 
                f_out.write(str(xoyo_out_f) + '\n')
                print('Fin=True; i_sol:', i_sol, ' con xoyo:',xoyolist[i_list-1], '\n')
                i_sol +=1
            
            #f_out.write('sign(r,b):'+str(np.sign(np.round([rlist[0],rlist[-1],blist[0],blist[-1]],5))).replace('\n', ',') + '\n')
            #f_out.write('sign(rlist,fixedparam):'+str(np.sign(np.round([rlist[0],fixedparameters[0:2],blist[0],fixedparameters[2:4]],5))).replace('\n', ',') + '\n')
            #### pruebas, corregir el \t por \n en la línea anterior
            #f_out.write('blist[-1]:'+str(blist[-1]).replace('\n', ',') + '\n')
            #f_out.write('fixedparam[2:4]'+str(fixedparameters[2:4]).replace('\n', ',') + '\n')
            f_out.close()

             
                

