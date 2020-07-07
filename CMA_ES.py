import numpy as np
from Functions import Function


class CMA_ES(Function):
    def __init__(self, target_func,fes):
        self.f = target_func
        self.lower = self.f.lower
        self.upper = self.f.upper
        self.dim = self.f.dimension
        #n = self.dim 
        self.fes = fes
        self.eval_times = 0 # not equivalent to g , g is number of generations , g = eval_times//lambda
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        #parameters setting(follow Table1)

        #Selection and Recombination

        #self.x_mean = np.random.uniform(self.lower,self.upper,self.dim)
        self.x_mean = np.random.random(self.dim)
        self.sigma = 0.3*(self.upper - self.lower)

        #Strategy parameter setting
        #Selection and Reombination
        self.lambda_ = 4 + int(3*np.log(self.dim))
        self.mu = int(self.lambda_/2)
        self.weights = np.array( [np.log(self.mu + 0.5) - np.log(i+1) for i in range(self.lambda_)]) #w'_i
        self.weights[:self.mu] = np.array([ w/sum(self.weights[:self.mu]) for w in self.weights[:self.mu] ]) # normalize  weights[:mu] , let sum(weights[:mu]) = 1
        self.mu_eff = 1/sum(np.power(w, 2) for w in self.weights[:self.mu])    #mu_eff = sum(i=1,mu,w'_i)**2 / sum(i=1,mu,w'_i**2)
        self.mu_eff_minus =  np.power(sum(self.weights[self.mu:]),2)/sum(np.power(w, 2) for w in self.weights[self.mu:])  #mu_eff_minus = sum(i=mu+1,lambda,w'_i)**2 / sum(i=mu+1,lambda,w'_i**2)

        #Covariance matrix adaptation
        self.alpha_cov = 2
        self.c_c = (4+self.mu_eff/self.dim)/(self.dim+4+2*self.mu_eff/self.dim) #c_c = (4+mu_eff/n)/(n+4+2*mu_eff/n) ; c_c is learning rate for p_c(p_c is evolution path for Covariance matrix)
        self.c_s = (self.mu_eff + 2)/(self.dim + self.mu_eff + 5) # c_s = (mu_eff+2)/(n+mu_eff+5) ; c_simga is learing rate for p_simga (p_simga is evolution path for step_size(simga) )
        self.c1 = self.alpha_cov/( (self.dim+1.3)**2 + self.mu_eff ) #c1 = alpha_cov/( (n+1.3)**2 + mu_eff ) ; c1 is learning rate for rank-one update of C
        self.c_mu = min([ 1-self.c1, self.alpha_cov*(self.mu_eff-2+1/self.mu_eff)/((self.dim+2)**2 + self.alpha_cov*self.mu_eff/2)]) #c_mu = min(1-c1,alpha_cov*(mu_eff-2+1/mu_eff)/( (n+2)**2 + alpha_cov*mu_eff/2) ) ;
        self.d_s = 1 + 2*max([0,  ((self.mu_eff-1)/(self.dim+1))**0.5 -1 ]) + self.c_s # d_s is damping factor for update of step_size(simga)

        #
        self.alpha_mu_minus = 1 + self.c1/self.c_mu
        self.alpha_mu_eff_minus = 1+(2*self.mu_eff_minus)/(self.mu_eff+2)
        self.alpha_posdef_minus = (1-self.c1-self.c_mu)/(self.dim*self.c_mu)
        #
        self.weights[self.mu:] = np.array([ w * min([self.alpha_mu_minus, self.alpha_mu_eff_minus, self.alpha_posdef_minus])/sum(abs(self.weights[self.mu:])) for w in self.weights[self.mu:] ])

        #Initialization
        self.p_c = np.zeros(self.dim) #evolution path for Covariance mextrix ; evolution path is a vector , its dimension is (n,1)
        self.p_s = np.zeros(self.dim) #evolution path for step_size(simga) ; evolution path is a vector , its dimension is (n,1)
        self.B = np.eye(self.dim) #dimension is (n,n)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim) # C= BD(BD)^t
        self.M = np.eye(self.dim)
        self.chiN = ((self.dim)**0.5)*(1-1/(4*self.dim)+1/(21*self.dim**2)) #E||N(0,I)|| = sqrt(2)*gamma_fun((n+1)/2)/gamma_fun(n/2) ~ sqrt(n)+O(1/n)  from eq(32)~(33)



    def step(self):
        #sample new population of search points for k=1,2,...,lambda
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        self.M = self.B * self.D

        z = np.zeros((self.lambda_, self.dim))
        y = np.zeros((self.lambda_, self.dim))
        x = np.zeros((self.lambda_, self.dim))
        fitvals = np.zeros(self.lambda_)

        for k in range(self.lambda_):
            if self.eval_times == self.fes:
                break

            print('=====================FE=====================')
            print(self.eval_times)
            z[k] = np.random.normal(0, 1, self.dim)
            y[k] = np.dot(self.M, z[k])
            x[k] = self.x_mean + self.sigma * y[k]
            x[k] = np.clip(x[k],self.lower,self.upper)
            print(x[k])
            fitvals[k] = self.f.evaluate(x[k])
            self.eval_times += 1

            if fitvals[k] < self.optimal_value:
                self.optimal_value = fitvals[k]
                self.optimal_solution = x[k]
            print("optimal: {}\n".format( self.get_optimal()[1]))

        if self.eval_times == self.fes:
            return
        #sort by fitness and update weighted mean
        argx = np.argsort(fitvals)
        old_x_mean = self.x_mean
        z_mean = np.sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        y_mean = np.sum(self.weights[i] * y[argx[i]] for i in range(self.mu))
        self.x_mean = self.x_mean + self.sigma * np.sum(self.weights[i] * y[argx[i]] for i in range(self.mu))

        #update evolution path
        self.p_s = (1-self.c_s)*self.p_s + np.sqrt(self.c_s*(2-self.c_s)*self.mu_eff)*np.dot(self.B,z_mean) #eq(43)  p_s(g+1) = (1-c_s)*p_s(g) + (c_s*(2-c_s)*mu_eff)**0.5 * y_w
        h_s = 1 if  ( np.linalg.norm(self.p_s)/np.sqrt(1-(1-self.c_s)**(2*(self.eval_times//self.lambda_))) ) < ( (1.4+2/(self.dim+1))*self.chiN ) else 0  #from A Algorithm Summary
        self.p_c = (1-self.c_c)*self.p_c + h_s*np.sqrt(self.c_c*(2-self.c_c)*self.mu_eff)*y_mean # eq(45) p_c(g+1) = (1-c_c)*p_c(g) + h_s*(c_c(2-c_c)*mu_eff)**0.5 *y_w

        # update covariance matrix
        delta_h = (1-h_s)*self.c_c*(2-self.c_c)
        part1 = (1+self.c1*delta_h-self.c1-self.c_mu)*self.C
        part2 = self.c1*np.dot(self.p_c.reshape(self.dim,1),self.p_c.reshape(1,self.dim))
        part3 = np.zeros((self.dim,self.dim))
        #eq(46)
        weights_o = self.weights.copy()
        for i in range(self.lambda_):
            if(weights_o[i]<0):
                weights_o[i] = weights_o[i]*self.dim/(np.linalg.norm(np.dot(self.B,z[argx[i]]))**2)
            part3 += self.c_mu*weights_o[i]*np.dot(y[argx[i]].reshape(self.dim, 1),y[argx[i]].reshape(1,self.dim))

        self.C = part1 + part2 + part3

        # update step-size
        self.sigma *= np.exp((self.c_s /self.d_s) * (np.linalg.norm(self.p_s)/self.chiN - 1))
        #self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

    def run(self):
        while self.eval_times < self.fes :
            self.step()

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value



