from Functions import Rosenbrock as rosen
from CMA_ES import CMA_ES as cmaes



if __name__ == '__main__':
    fes = 3000
    func = rosen(5,-50,50)
    opt = cmaes(func,fes)
    opt.run()
    print("optimal_solution : {}\n".format( opt.get_optimal()[0]))
    print("optimal_value : {}\n".format( opt.get_optimal()[1]))