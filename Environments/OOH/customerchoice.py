from math import exp
from numpy.random import gumbel
import numpy as np

class customerchoicemodel(object):
    def __init__(self,
                 base_util,
                 dist_scaler,
                 euclidean,
                 dist_mat,
                 n_cust,
                 external_option=False,
                 external_base_util=0.0,
                 external_price_sensitivity=0.0,
                 external_price=0.0):
        self.euclidean_distance = euclidean
        self.dist_scaler = dist_scaler
        self.base_util = base_util
        self.dist_mat = dist_mat
        self.n_cust = n_cust
        self.external_option = bool(external_option)
        self.external_base_util = float(external_base_util)
        self.external_price_sensitivity = float(external_price_sensitivity)
        self.external_price = float(external_price)
        if len(self.dist_mat)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid
        
    def external_utility(self):
        """Outside option utility: U_external = base_external - alpha * price."""
        return self.external_base_util - self.external_price_sensitivity * self.external_price
        
    def mnl_euclid(self,customer,parcelpoint):
        """
        multi-nomial logit model calculating euclidean distance
        """
        distance = self.euclidean_distance(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = -exp(-distance/self.dist_scaler)
        return self.base_util + beta_p

    def mnl_distmat(self,customer,parcelpoint):
        """
        multi-nomial logit model using distance matrix
        """
        distance = self.dist_mat[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = -exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def customerchoice_offer(self,customer,action,parcelpoints):
        """
        Customer choice model for the offering decision, i.e., action is 1 parcelpoint offer.
        The optional external option lets the customer leave without choosing home or a parcelpoint.
        """
        action = np.asarray(action, dtype=int)
        pps = parcelpoints[action-self.n_cust]
        n_internal = len(action)+1
        shape = (n_internal + int(self.external_option), 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        if self.external_option:
            utils[-1] = self.external_utility()
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)
        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, 0#home delivery
        elif self.external_option and idx == n_internal:
            return None, False, -2, 0.0#external option / customer exits
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,0#accept offer

    def customerchoice_pricing(self,customer,action,parcelpoints):
        """
        Customer choice model for the pricing decision, i.e., action is vector of prices for all PPs and home delivery.
        The external option is not priced by the platform and can be selected with probability > 0.
        """
        action = np.asarray(action, dtype=float).reshape(-1)
        pps = parcelpoints[parcelpoints.mask].data
        n_internal = len(pps)+1
        shape = (n_internal + int(self.external_option), 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils[:len(action)] = np.add(utils[:len(action)],customer.incentiveSensitivity*action.reshape((len(action),1)))#incentive
        if self.external_option:
            utils[-1] = self.external_utility()
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, action[0]#home delivery
        elif self.external_option and idx == n_internal:
            return None, False, -2, 0.0#external option / customer exits
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,action[idx]#accept offer
