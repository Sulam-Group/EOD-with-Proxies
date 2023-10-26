import numpy as np
import scipy as sp
import utils

class BinaryBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,
                 a_hat,
                 adjusted = True):
            
        # Setting the variables
        self.y = y
        self.y_ = y_
        self.a = a
        self.a_hat = a_hat

        '''
        Calculating metrics for Y wrt to A
        '''

        # Calculating P(A = 0) and P(A = 1)
        self.a_groups = np.unique(a)
        a_group_ids = [np.where(a == g)[0] for g in self.a_groups]
        self.p_a = [len(cols)/len(a) for cols in a_group_ids]

        # Calcuating P(A,Y)
        self.a_gr_list = [utils.CLFRates(self.y[i], self.y_[i]) for i in a_group_ids]
        self.a_group_rates = dict(zip(self.a_groups, self.a_gr_list))
        self.base_rates = {'r_11': np.sum(y[a==1] == 1)/len(y),
                           'r_01': np.sum(y[a==0] == 1)/len(y),
                           'r_10': np.sum(y[a==1] == 0)/len(y),
                           'r_00': np.sum(y[a==0] == 0)/len(y)}

        '''
        Calculating metrics for Y wrt to A_hat
        '''

        # Calculating P(A_hat=0) and P(A_hat=1)
        self.a_hat_groups = np.unique(a_hat)
        a_hat_group_ids = [np.where(a_hat == g)[0] for g in self.a_hat_groups]
        self.p_a_hat = [len(cols)/len(a) for cols in a_hat_group_ids]

        # Calcuating P(A_hat, Y)
        self.a_hat_gr_list = [utils.CLFRates(self.y[i], self.y_[i]) for i in a_hat_group_ids]
        self.a_hat_group_rates = dict(zip(self.a_hat_groups, self.a_hat_gr_list))
        self.est_base_rates = {'rh_11': np.sum(y[a_hat==1] == 1)/len(y),
                               'rh_01': np.sum(y[a_hat==0] == 1)/len(y),
                               'rh_10': np.sum(y[a_hat==1] == 0)/len(y),
                               'rh_00': np.sum(y[a_hat==0] == 0)/len(y)}
        
        '''
        Calculating remaining metrics
        '''

        # Overall rates for Y and Y_hat
        self.overall_rates = utils.CLFRates(self.y, self.y_)

        # Remaining relevant variables
        # self.a_hat_rates = updated_tools.CLFRates(self.a, self.a_hat)
        U0 = np.sum(a_hat[a==1] == 0)/len(a)
        U1 = np.sum(a_hat[a==0] == 1)/len(a)
        self.U0 = U0
        self.U1 = U1

        # True group conditional TPRs and FPRs 
        alpha_11, alpha_01 = self.a_group_rates[1].tpr, self.a_group_rates[0].tpr
        alpha_10, alpha_00 = self.a_group_rates[1].fpr, self.a_group_rates[0].fpr

        # Estimated group conditional TPRs and FPRs
        alpha_h_11, alpha_h_01 = self.a_hat_group_rates[1].tpr, self.a_hat_group_rates[0].tpr
        alpha_h_10, alpha_h_00 = self.a_hat_group_rates[1].fpr, self.a_hat_group_rates[0].fpr

        # Instantiating estimated base rates
        rh_11, rh_01 = self.est_base_rates['rh_11'], self.est_base_rates['rh_01']
        rh_10, rh_00 = self.est_base_rates['rh_10'], self.est_base_rates['rh_00']

        # c constants (in paper)
        self.c_01 = rh_01 + U1 - U0
        self.c_00 = rh_00 + U1 - U0

        # k constants (in paper)
        self.k_11 = rh_11 + U0 - U1
        self.k_10 = rh_10 + U0 - U1

        # Calculate true delta tpr and fpr
        self.d_tpr = alpha_11 - alpha_01
        self.d_fpr = alpha_10 - alpha_00

        # Calculate initial upper and lower bounds
        bounds = {
            'tpr': {'lb': (rh_11/self.k_11)*alpha_h_11 - (rh_01/self.c_01)*alpha_h_01 - U1*(1/self.k_11 + 1/self.c_01),
                    'ub': (rh_11/self.k_11)*alpha_h_11 - (rh_01/self.c_01)*alpha_h_01 + U0*(1/self.k_11 + 1/self.c_01)},
            'fpr': {'lb': (rh_10/self.k_10)*alpha_h_10 - (rh_00/self.c_00)*alpha_h_00 - U1*(1/self.k_10 + 1/self.c_00),
                    'ub': (rh_10/self.k_10)*alpha_h_10 - (rh_00/self.c_00)*alpha_h_00 + U0*(1/self.k_10 + 1/self.c_00)},
        }

        assumption = 1
        tpr_assumption = 1
        fpr_assumption = 1
        if adjusted == False:
        # Determine where the assumption does not hold and replace bounds with correct ones 
            if not (U1/rh_11 <= alpha_h_11 <= 1-(U1/rh_11)):
                bounds['tpr']['ub'] = 1 - (alpha_h_01*rh_01 - U0)/(rh_11 + rh_01 - alpha_h_11*rh_11 - U0)
                assumption = 0
                tpr_assumption = 0
            if not (U0/rh_01 <= alpha_h_01 <= 1-(U0/rh_01)):
                bounds['tpr']['lb'] = (alpha_h_11*rh_11 - U1)/(rh_11 + rh_01 - alpha_h_01*rh_01 - U1) - 1
                assumption = 0
                tpr_assumption = 0
            if not (U1/rh_10 <= alpha_h_10 <= 1-(U1/rh_11)):
                bounds['fpr']['lb'] = -(alpha_h_10*rh_10 + alpha_h_00*rh_00)/(alpha_h_10*rh_10 + rh_00 - U0)
                assumption = 0
                fpr_assumption = 0
            if not (U0/rh_00 <= alpha_h_00 <= 1-(U0/rh_00)):
                bounds['fpr']['ub'] = (alpha_h_10*rh_10 + alpha_h_00*rh_00)/(alpha_h_00*rh_00 + rh_10 - U1)
                assumption = 0
                fpr_assumption = 0
            self.bounds = bounds
            self.assumption = assumption
            self.tpr_assumption = tpr_assumption
            self.fpr_assumption = fpr_assumption
        else:
            self.bounds = bounds
            self.assumption = assumption
            self.tpr_assumption = tpr_assumption
            self.fpr_assumption = fpr_assumption

    def adjust(self,
               con='tpr/fpr',
               obj='opt',
               imbalanced=True,
               binom=False):
        
        # Instantiating relevant parameters
        rh_11, rh_01 = self.est_base_rates['rh_11'], self.est_base_rates['rh_01']
        rh_10, rh_00 = self.est_base_rates['rh_10'], self.est_base_rates['rh_00']
        rh_11 = self.est_base_rates['rh_11']
        c_00, c_01 = self.c_00, self.c_01
        k_10, k_11 = self.k_10, self.k_11
        U0, U1 = self.U0, self.U1

        # Setting loss 
        if imbalanced == True:
            l_10 = 0.5*(1/(self.overall_rates.num_neg))
            l_01 = 0.5*(1/(self.overall_rates.num_pos))
        else:
            l_10 = 1
            l_01 = 1

        # Getting the coefficients for the linear program
        coefs = [((l_10 * g.tnr * g.num_neg - l_01 * g.fnr * g.num_pos)*self.p_a_hat[i],
               (l_10 * g.fpr * g.num_neg - l_01 * g.tpr * g.num_pos)*self.p_a_hat[i])
               for i, g in enumerate(self.a_hat_gr_list)]
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.zeros((12))
        obj_coefs[:4] = np.array(coefs).flatten()
        obj_bounds = [(0, 1)]

        # # Constraint matrix and vector for generalized linear program
        g0 = self.a_hat_gr_list[0]
        g1 = self.a_hat_gr_list[1]

        # Matrix A in Ax = b
        A_opt = np.zeros((10,12))
        A_opt[0,0] = (rh_00/c_00)*(1 - g0.fpr)
        A_opt[0,1] = (rh_00/c_00)*g0.fpr
        A_opt[0,2] = -(rh_10/k_10)*(1-g1.fpr)
        A_opt[0,3] = -(rh_10/k_10)*g1.fpr
        A_opt[1,0] = (rh_01/c_01)*(1 - g0.tpr)
        A_opt[1,1] = (rh_01/c_01)*g0.tpr
        A_opt[1,2] = -(rh_11/k_11)*(1-g1.tpr)
        A_opt[1,3] = -(rh_11/k_11)*g1.tpr
        A_opt[2,0], A_opt[2,1], A_opt[2,4] = (1 - g0.fpr), g0.fpr, -1
        A_opt[3,0], A_opt[3,1], A_opt[3,5] = (1 - g0.fpr), g0.fpr, 1
        A_opt[4,2], A_opt[4,3], A_opt[4,6] = (1-g1.fpr), g1.fpr, -1
        A_opt[5,2], A_opt[5,3], A_opt[5,7] = (1-g1.fpr), g1.fpr, 1
        A_opt[6,0], A_opt[6,1], A_opt[6,8] = (1 - g0.tpr), g0.tpr, -1
        A_opt[7,0], A_opt[7,1], A_opt[7,9] = (1 - g0.tpr), g0.tpr, 1
        A_opt[8,2], A_opt[8,3], A_opt[8,10] = (1-g1.tpr), g1.tpr, -1
        A_opt[9,2], A_opt[9,3], A_opt[9,11] = (1-g1.tpr), g1.tpr, 1

        # Vector b in Ax = b
        b_opt = np.zeros(A_opt.shape[0])
        b_opt[0], b_opt[1] = (0.5*(U0-U1))*(1/k_10 + 1/c_00), (0.5*(U0-U1))*(1/k_11 + 1/c_01)
        b_opt[2], b_opt[3] = U0/rh_00, 1 - (U0/rh_00)
        b_opt[4], b_opt[5] = U1/rh_10, 1 - (U1/rh_10)
        b_opt[6], b_opt[7] = U0/rh_01, 1 - (U0/rh_01)
        b_opt[8], b_opt[9] = U1/rh_11, 1 - (U1/rh_11)

        # Constraint matrix and vector for fairness correction wrt A_hat

        # Matrix A in Ax = b
        A = np.zeros((10,12))
        A[0,0] = (1 - g0.fpr)
        A[0,1] = g0.fpr
        A[0,2] = -(1-g1.fpr)
        A[0,3] = -g1.fpr
        A[1,0] = (1 - g0.tpr)
        A[1,1] = g0.tpr
        A[1,2] = -(1-g1.tpr)
        A[1,3] = -g1.tpr
        A[2,0], A[2,1], A[2,4] = (1 - g0.fpr), g0.fpr, -1
        A[3,0], A[3,1], A[3,5] = (1 - g0.fpr), g0.fpr, 1
        A[4,2], A[4,3], A[4,6] = (1-g1.fpr), g1.fpr, -1
        A[5,2], A[5,3], A[5,7] = (1-g1.fpr), g1.fpr, 1
        A[6,0], A[6,1], A[6,8] = (1 - g0.tpr), g0.tpr, -1
        A[7,0], A[7,1], A[7,9] = (1 - g0.tpr), g0.tpr, 1
        A[8,2], A[8,3], A[8,10] = (1-g1.tpr), g1.tpr, -1
        A[9,2], A[9,3], A[9,11] = (1-g1.tpr), g1.tpr, 1

        # Vector b in Ax = b
        b = np.zeros(A.shape[0])
        b[0], b[1] = 0, 0
        b[2], b[3] = U0/rh_00, 1 - (U0/rh_00)
        b[4], b[5] = U1/rh_10, 1 - (U1/rh_10)
        b[6], b[7] = U0/rh_01, 1 - (U0/rh_01)
        b[8], b[9] = U1/rh_11, 1 - (U1/rh_11)
        
        if con == 'tpr/fpr':
            if obj == 'opt':
                self.con_A = A_opt
                self.con_b = b_opt
            elif obj == 'fair':
                self.con_A = A
                self.con_b = b
            elif obj == 'project':
                indices = [2,3,4,5,6,7,8,9]
                self.con_A = A[indices,:]
                self.con_b = b[indices]
        
        elif con == 'tpr':
            indices = [1,6,7,8,9]
            if obj == 'opt':
                self.con_A = A_opt[indices,:]
                self.con_b = b_opt[indices]
            elif obj == 'fair':
                self.con_A = A[indices,:]
                self.con_b = b[indices]
            elif obj == 'project':
                indices = [6,7,8,9]
                self.con_A = A[indices,:]
                self.con_b = b[indices]
        
        elif con == 'fpr':
            indices = [0,2,3,4,5]
            if obj == 'opt':
                self.con_A = A_opt[indices,:]
                self.con_b = b_opt[indices]
            elif obj == 'fair':
                self.con_A = A[indices,:]
                self.con_b = b[indices]
            elif obj == 'project':
                indices = [2,3,4,5]
                self.con_A = A[indices,:]
                self.con_b = b[indices]

        else:
            print('constraint not specified!')


        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=self.con_A,
                                       b_eq=self.con_b,
                                       method='highs')
        self.pya = self.opt.x[:4].reshape(len(self.a_hat_groups), 2)
        
        '''
        Calculating new theoretical parameters
        '''
        self.new_a_hat_group_rates = {
            '0': {'tpr': (1-g0.tpr)*self.opt.x[0] + g0.tpr*self.opt.x[1], 'fpr': (1-g0.fpr)*self.opt.x[0] + g0.fpr*self.opt.x[1]},
            '1': {'tpr': (1-g1.tpr)*self.opt.x[2] + g1.tpr*self.opt.x[3], 'fpr': (1-g1.fpr)*self.opt.x[2] + g1.fpr*self.opt.x[3]},
        }
        
        self.new_bounds = {
            'tpr': {'lb': (rh_11/k_11)*self.new_a_hat_group_rates['1']['tpr'] - (rh_01/c_01)*self.new_a_hat_group_rates['0']['tpr'] - U1*(1/k_11 + 1/c_01),
                    'ub': (rh_11/k_11)*self.new_a_hat_group_rates['1']['tpr'] - (rh_01/c_01)*self.new_a_hat_group_rates['0']['tpr'] + U0*(1/k_11 + 1/c_01)},
            'fpr': {'lb': (rh_10/k_10)*self.new_a_hat_group_rates['1']['fpr'] - (rh_00/c_00)*self.new_a_hat_group_rates['0']['fpr'] - U1*(1/k_10 + 1/c_00),
                    'ub': (rh_10/k_10)*self.new_a_hat_group_rates['1']['fpr'] - (rh_00/c_00)*self.new_a_hat_group_rates['0']['fpr'] + U0*(1/k_10 + 1/c_00)},
        }

        # Calculating adjusted predictions
        self.y_adj = utils.pred_from_pya(y_=self.y_, 
                                         a=self.a_hat,
                                         pya=self.pya, 
                                         binom=binom)