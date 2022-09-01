import torch
import numpy as np
# import signal_tl as stl

import sys
import rtamt
"""
Runway R2 ( not fully updated)
Spec 1: phi_1 = downwind: x [0.5, 3.0]
						  y [-2.5, -0.8]
						  z [0.5, 0.7]

Spec 2: phi_2 = base:     x [2, 4.0]
						  y [-1.8, 0.02]
						  z [0.3, 0.5]

Spec 3: phi_3 = final     x [1.7, 3.7]
						  y [-0.08, 0.08]
						  x [0.3, 0.5]

Spec 4: psi = overflying  x [0.0, 1.45]
						  y [-0.01, 0.01]
						  z [0.1, 0.6]

"""

phi_1_x = [0.5, 2.50]
phi_1_y = [-2.5, -1.0]
phi_1_z = [0.5, 0.7]
phi_2_x = [2.5, 4.0]
phi_2_y = [-2.5, -0.5] 
phi_2_z = [0.3, 0.5]
phi_3_x = [1.5, 4.0]# region 3 isolated for testing purposes
phi_3_y = [-0.5, 0.2]
phi_3_z = [0.3, 0.5]

psi_x = [0.0, 1.45]
psi_y = [-0.01, 0.01]
psi_z = [0.1, 0.6]


def monitor_R2(ways): # rtamt specs for runway R2
	# data
	# print("Hi", trajs0.shape)
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]*1.1).tolist()
	trajs_z = (ways[:,2]*1.2).tolist()
	r2 = np.array([[1.48,0.0,0.256]])
	region1 = np.array([[1.5,1.75,0.84]])
	region2 = np.array([[3.25,1.75,0.6]])

	trajs_new_diff = np.diff(ways, axis=0)
	# diff_tan = np.arctan2(trajs_new_diff[:,1],trajs_new_diff[:,0])
	# diff_tan = diff_tan*180.0/np.pi
	# diff_tan_list = diff_tan.tolist()
	# print(diff_tan_list)
	if torch.is_tensor(ways):
		ways = ways.numpy()
	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], r2.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(r2, 2), 1).reshape(r2.shape[0], 1), (r2.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_runway = np.sqrt(sqrA - 2*np.matmul(ways, r2.transpose()) + sqrB)
	L2_dist_runway = L2_dist_runway*0.1
	L2_dist_runway = L2_dist_runway.squeeze().tolist()
	diff_tan = np.arctan2(trajs_new_diff[:,1],trajs_new_diff[:,0])
	# if 0.3490 > diff_tan[-1] >-0.3490:
	# 	print("diff_tan",diff_tan[-1])
	# diff_tan = diff_tan*180.0/np.pi
	# diff_tan_list2 = diff_tan.tolist()
	# diff_tan_list2.append(diff_tan_list2[-1])
	# print("diff_tan_list",diff_tan_list2)
	diff_tan = np.concatenate((diff_tan, diff_tan[-1:]),axis=0)
	diff_tan[diff_tan < -1.57] += 2 * np.pi

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region1.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region1, 2), 1).reshape(region1.shape[0], 1), (region1.shape[0], ways.shape[0])).transpose()
	#
	region_1_dist = np.sqrt(sqrA - 2*np.matmul(ways, region1.transpose()) + sqrB)

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region2.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region2, 2), 1).reshape(region2.shape[0], 1), (region2.shape[0], ways.shape[0])).transpose()
	#
	region_2_dist = np.sqrt(sqrA - 2*np.matmul(ways, region2.transpose()) + sqrB)
	# diff_tan_list = [(2*np.pi + i) if i < -1.57 else i for i in diff_tan_list2]
	# diff_tan_list= diff_tan_list2
	# print("diff_tan_list",diff_tan_list)
	phi_1l =  (np.array(diff_tan) + 0.3490)*1.0
	phi_1u =  (0.3490 - np.array(diff_tan))*1.0

	phi_2l =  (np.array(diff_tan) - 1.0472)*0.5
	phi_2u =  (2.0944 - np.array(diff_tan))*0.5

	phi_3l =  np.array(diff_tan) - 2.8798
	phi_3u =  3.4034 - np.array(diff_tan)
	# print("trajs_x",trajs_x.shape)
	# print(trajs_y)
	# print("phi_1u",max(phi_1u))
	trajs_vel_diff = np.diff(ways, axis=0)
	# print(trajs_vel_diff)
	diff_vel = np.concatenate((trajs_vel_diff, trajs_vel_diff[-1:]),axis=0)
	# print("diff_vel",diff_vel)
	vels_x = diff_vel[:,0].tolist()
	vels_y = diff_vel[:,1].tolist()
	vels_z = diff_vel[:,2].tolist()


	# user_details = [{'name' : x, 'rank' : trajs_x.index(x)} for x in trajs_x]
	trajs_x = list(enumerate(trajs_x))
	trajs_y = list(enumerate(trajs_y))
	trajs_z = list(enumerate(trajs_z))

	diff_tan_list = list(enumerate(diff_tan))
	phi_1l = list(enumerate(phi_1l))
	phi_1u = list(enumerate(phi_1u))
	phi_2l = list(enumerate(phi_2l))
	phi_2u = list(enumerate(phi_2u))
	phi_3l = list(enumerate(phi_3l))
	phi_3u = list(enumerate(phi_3u))
	vel_x = list(enumerate(vels_x))
	vel_y = list(enumerate(vels_y))
	vel_z = list(enumerate(vels_z))

	L2_runway = list(enumerate(L2_dist_runway))
	L2_region1 = list(enumerate(region_1_dist))
	L2_region2 = list(enumerate(region_2_dist))


	# # stl
	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	# spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
	spec.name = 'runway r2'
	spec.declare_const('threshold_1', 'float', '3')

	spec.declare_const('threshold_2', 'float', '5')
	spec.declare_var('x', 'float')
	spec.declare_var('iq', 'float')
	spec.declare_var('up', 'float')
	# spec.declare_var('xl', 'float')
	# spec.declare_var('xu', 'float')
	spec.declare_var('y', 'float')
	# spec.declare_var('yl', 'float')
	# spec.declare_var('yu', 'float')
	spec.declare_var('z', 'float')
	# spec.declare_var('zl', 'float')
	# spec.declare_var('zu', 'float')

	spec.declare_var('diff_tan', 'float')
	spec.declare_var('phi_1u_head', 'float')
	spec.declare_var('phi_1l_head', 'float')

	spec.declare_var('phi_2u_head', 'float')
	spec.declare_var('phi_2l_head', 'float')

	spec.declare_var('phi_3u_head', 'float')
	spec.declare_var('phi_3l_head', 'float')
	spec.declare_var('out', 'float')

	spec.declare_var('final', 'float')
	spec.declare_var('downwind', 'float')
	spec.declare_var('base', 'float')
	spec.declare_var('vel_x', 'float')
	spec.declare_var('vel_y', 'float')
	spec.declare_var('vel_z', 'float')
	# spec.declare_var('out', 'float')
	# spec.set_var_io_type('xl', 'input')
	# spec.set_var_io_type('xu', 'input')

	spec.spec = 'eventually( (( (x >-1.5) and (x < 1.5)) and ( (y > -3) and (y < -2) )   and (z<0.5) ) and (eventually( (((x > 4.5) and (x < 5.0) ) and ((y > -3) and (y < 0.2) and (z<0.5))  ) and (eventually( ((x > 1.3) and (x < 1.5) ) and ((y > -0.2) and (y < 0.2) and z<0.5 ) )))))'
	# spec.spec = 'eventually( (( (x >-1.5) and (x < 1.5)) and ( (y > -3) and (y < -2) )    ) and eventually( ((x > 4.5) and (x < 5.0) ) and ((y > -3) and (y < 0.2)) ))'



	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['x',trajs_x],['y',trajs_y],['z',trajs_z],['vel_x',vel_x],['vel_y',vel_y],['vel_z',vel_z],['final',L2_runway],['downwind',L2_region1],['base',L2_region2],['phi_1l_head',phi_1l],['phi_1u_head',phi_1u],['phi_2l_head',phi_2l],['phi_2u_head',phi_2u],['phi_3l_head',phi_3l],['phi_3u_head',phi_3u ])

	# out = spec.evaluate(['x', trajs_x],['y', trajs_y],['diff_tan',diff_tan] ,['phi_l',phi_l],['phi_u',phi_u],['phi_l2',phi_l2],['phi_u2',phi_u2],['phi_l3',phi_l3],['phi_u3',phi_u3],['vel_x',vel_x],['vel_y',vel_y],['vel_z',vel_z])
	# out = spec.evaluate(['x', trajs_x],['y', trajs_y],['z', trajs_z],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z])
	# print('Robustness offline: {}'.format(out))
	return out[0][1]


# def monitor_R1(trajs0):
# 	# data
# 	trajs_x = trajs0[:,0].tolist()
# 	trajs_y = trajs0[:,1].tolist()
# 	trajs_z = trajs0[:,2].tolist()
# 	x = lambda a, b : a * b
# 	# user_details = [{'name' : x, 'rank' : trajs_x.index(x)} for x in trajs_x]
# 	trajs_x = list(enumerate(trajs_x))
# 	trajs_y = list(enumerate(trajs_y))
# 	trajs_z = list(enumerate(trajs_z))
# 	# print(trajs_x)
# 	# # stl
# 	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
# 	# spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
# 	spec.name = 'runway r1'
# 	spec.declare_var('x', 'float')
# 	spec.declare_var('xl', 'float')
# 	spec.declare_var('xu', 'float')

# 	spec.declare_var('yl', 'float')
# 	spec.declare_var('yu', 'float')

# 	spec.declare_var('zl', 'float')
# 	spec.declare_var('zu', 'float')

# 	# spec.declare_var('out', 'float')
# 	spec.set_var_io_type('xl', 'input')
# 	spec.set_var_io_type('xu', 'input')

# 	spec.set_var_io_type('yl', 'input')
# 	spec.set_var_io_type('yu', 'input')

# 	spec.set_var_io_type('zl', 'input')
# 	spec.set_var_io_type('zu', 'input')

# #     spec.declare_var('phi_1', 'float')
# #     spec.declare_var('phi_2', 'float')
# 	#spec.declare_var('phi_3', 'float')
# 	# spec.set_var_io_type('out', 'output')
# 	# spec.spec = '(eventually[0:0] ((always[0:0] (xl > 0.5)) and (always[0:0] (xu < 3.0))) )'
# 	#spec.add_sub_spec = ('phi_1 = (eventually[0:0] (always[0:0]( (xl >-1.2) and (xu < 0.9) ) and ( (yl > 0.8) and (yu < 2.5) ) and ( (zl > 0.5) and (zu < 0.7) ) ) )')
# 	#spec.add_sub_spec = ('phi_2 = (eventually[0:0] (always[0:0]( (xl > -2) and (xu < -1.2) ) and ( (yl > 0.08) and (yu < 2.5) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')
# 	#spec.add_sub_spec = ('phi_3 = (eventually[0:0] (always[0:0]( (xl > -3.0) and (xu < 0.0) ) and ( (yl > -0.08) and (yu < 0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')
# 	#spec.add_sub_spec = ('phi_1 = ( ( (xl >-1.2) and (xu < 0.9) ) and ( (yl > 0.8) and (yu < 2.5) ) and ( (zl > 0.5) and (zu < 0.7) ) ) ')
# 	#spec.add_sub_spec = ('phi_2 = ( ( (xl > -2) and (xu < -1.2) ) and ( (yl > 0.08) and (yu < 2.5) ) and ( (zl > 0.3) and (zu < 0.5) ) )')
# 	#spec.add_sub_spec = ('phi_3 = (eventually[0:0] (always[0:0]( (xl > -3.0) and (xu < 0.0) ) and ( (yl > -0.08) and (yu < 0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')

# 	# spec.spec = 'eventually(( ( (xl >-1.2) and (xu < 0.9) ) and ( (yl > 0.8) and (yu < 2.5) ) and ( (zl > 0.5) and (zu < 0.7) ) ) and eventually( ( (xl > -2) and (xu < -1.2) ) and ( (yl > 0.08) and (yu < 2.5) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )'
# 	spec.spec ='eventually( ( (x >0.5) and (x < 2.5))  and eventually( ((x > 2.5) and (x < 4.0)   ) ))'

# 	try:
# 		spec.parse()
# 		# spec.pastify()
# 	except rtamt.STLParseException as err:
# 		print('STL Parse Exception: {}'.format(err))
# 		sys.exit()


# 	rob = spec.evaluate(['x', trajs_x],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z])
# 	# print('Robustness offline: {}'.format(rob),rob[0])
# 	return rob[0][1]


# monitor_R1()






def monitor_R1():
	pass
#     # data

#     # # stl
#     spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
#     # spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
#     spec.name = 'runway r1'
#     spec.declare_var('xl', 'float')
#     spec.declare_var('xu', 'float')

#     spec.declare_var('yl', 'float')
#     spec.declare_var('yu', 'float')

#     spec.declare_var('zl', 'float')
#     spec.declare_var('zu', 'float')

#     # spec.declare_var('out', 'float')
#     spec.set_var_io_type('xl', 'input')
#     spec.set_var_io_type('xu', 'input')

#     spec.set_var_io_type('yl', 'input')
#     spec.set_var_io_type('yu', 'input')

#     spec.set_var_io_type('zl', 'input')
#     spec.set_var_io_type('zu', 'input')

#     spec.declare_var('phi_1', 'float')
#     spec.declare_var('phi_2', 'float')
#     spec.declare_var('phi_3', 'float')
#     # spec.set_var_io_type('out', 'output')
#     # spec.spec = '(eventually[0:0] ((always[0:0] (xl > 0.5)) and (always[0:0] (xu < 3.0))) )'
#     spec.add_sub_spec = ('phi_1 = (eventually[0:0] (always[0:0]( (xl >-1.2) and (xu < 0.9) ) and ( (yl > 0.8) and (yu < 2.5) ) and ( (zl > 0.5) and (zu < 0.7) ) ) )')
#     spec.add_sub_spec = ('phi_2 = (eventually[0:0] (always[0:0]( (xl > -2) and (xu < -1.2) ) and ( (yl > 0.08) and (yu < 2.5) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')
#     spec.add_sub_spec = ('phi_3 = (eventually[0:0] (always[0:0]( (xl > -3.0) and (xu < 0.0) ) and ( (yl > -0.08) and (yu < 0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')

#     spec.spec = '( ((phi_1) until[0:0] (phi_2)) until[0:0] (phi_3) )'
#     try:
#         spec.parse()
#         spec.pastify()
#     except rtamt.STLParseException as err:
#         print('STL Parse Exception: {}'.format(err))
#         sys.exit()


#     rob = spec.evaluate(['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z])
#     print('Robustness offline: {}'.format(rob))
#     return rob






#########################







# class MCTS_STL_spec_R2:
# #     def __init__(self):
# 	xl = stl.Predicate("xl") > phi_1_x[0]
# 	xu = stl.Predicate("xu") < phi_1_x[1]

# 	yl = stl.Predicate("yl") > phi_1_y[0]
# 	yu = stl.Predicate("yu") < phi_1_y[1]

# 	zl = stl.Predicate("zl") > phi_1_z[0]
# 	zu = stl.Predicate("zu") < phi_1_z[1]
# 	phi_1 = stl.Eventually(stl.Always((xl&xu)&(yl&yu)&(zl&zu)))


# 	xl2 = stl.Predicate("xl") > phi_2_x[0]
# 	xu2 = stl.Predicate("xu") < phi_2_x[1]

# 	yl2 = stl.Predicate("yl") > phi_2_y[0]
# 	yu2 = stl.Predicate("yu") < phi_2_y[1]

# 	zl2 = stl.Predicate("zl") > phi_2_z[0]
# 	zu2 = stl.Predicate("zu") < phi_2_z[1]
# 	phi_2 = stl.Eventually(stl.Always((xl2&xu2)&(yl2&yu2)&(zl2&zu2)))



# 	xl3 = stl.Predicate("xl") > phi_3_x[0]
# 	xu3 = stl.Predicate("xu") < phi_3_x[1]

# 	yl3 = stl.Predicate("yl") > phi_3_y[0]
# 	yu3 = stl.Predicate("yu") < phi_3_y[1]

# 	zl3 = stl.Predicate("zl") >phi_3_z[0]
# 	zu3 = stl.Predicate("zu") < phi_3_z[1]
# 	phi_3 = stl.Eventually(stl.Always((xl3&xu3)&(yl3&yu3)&(zl3&zu3)))

# 	phi_1_new = stl.Eventually((xl&xu)&(yl&yu)&(zl&zu) & stl.Eventually((xl2&xu2)&(yl2&yu2)&(zl2&zu2)))
# 	phi_123_new= stl.Eventually(phi_1_new & stl.Eventually((xl3&xu3)&(yl3&yu3)&(zl3&zu3)))
# 	# phi_1_2 = stl.Until(phi_1,phi_2)
# 	phi_1_2 = phi_1 & stl.Eventually(phi_2)
# 	phi_1_2_3 = stl.Until(phi_1_2,phi_3)


# 	xl4 = stl.Predicate("xl") > psi_x[0]
# 	xu4 = stl.Predicate("xu") < psi_x[1]

# 	yl4 = stl.Predicate("yl") > psi_y[0]
# 	yu4 = stl.Predicate("yu") < psi_y[1]

# 	zl4 = stl.Predicate("zl") > psi_z[0]
# 	zu4 = stl.Predicate("zu") < psi_z[1]
# 	psi = stl.Always(stl.Not((xl4&xu4)&(yl4&yu4)&(zl4&zu4)))


# 	# phi = psi & phi_1_2_3
# 	phi = psi & phi_1_2_3

# 	@staticmethod
# 	def evaluate(trajectory: torch.Tensor, step_number: int, agent) -> float:
# 		with torch.no_grad():

# 			x_traj = trajectory[:, 0].tolist()
# 			y_traj = trajectory[:, 1].tolist()
# 			z_traj = trajectory[:, 2].tolist()
# 			t = np.arange(step_number, step_number + int(trajectory.shape[0])).tolist()
# 			trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
# 			rob = stl.compute_robustness(MCTS_STL_spec_R2.phi_123_new, trace, synchronized=True)
# 			return rob
			
# phi_1_x_r1 = [-1.2, 0.9]
# phi_1_y_r1 = [0.8, 2.5]
# phi_1_z_r1 = [0.5, 0.7]
# phi_2_x_r1 = [-3, -1.2]
# phi_2_y_r1 = [0.08, 2.5]
# phi_2_z_r1 = [0.3, 0.5]
# phi_3_x_r1 = [-3.0, 0.0]
# phi_3_y_r1 = [-0.08,0.08]
# phi_3_z_r1 = [0.3, 0.5]

# class MCTS_STL_spec_R1:
# #     def __init__(self):
# 	xl = stl.Predicate("xl") > phi_1_x_r1[0]
# 	xu = stl.Predicate("xu") < phi_1_x_r1[1]

# 	yl = stl.Predicate("yl") > phi_1_y_r1[0]
# 	yu = stl.Predicate("yu") < phi_1_y_r1[1]

# 	zl = stl.Predicate("zl") > phi_1_z_r1[0]
# 	zu = stl.Predicate("zu") < phi_1_z_r1[1]
# 	phi_1 = stl.Eventually(stl.Always((xl&xu)&(yl&yu)&(zl&zu)))


# 	xl2 = stl.Predicate("xl") > phi_2_x_r1[0]
# 	xu2 = stl.Predicate("xu") < phi_2_x_r1[1]

# 	yl2 = stl.Predicate("yl") > phi_2_y_r1[0]
# 	yu2 = stl.Predicate("yu") < phi_2_y_r1[1]

# 	zl2 = stl.Predicate("zl") > phi_2_z_r1[0]
# 	zu2 = stl.Predicate("zu") < phi_2_z_r1[1]
# 	phi_2 = stl.Eventually(stl.Always((xl2&xu2)&(yl2&yu2)&(zl2&zu2)))



# 	xl3 = stl.Predicate("xl") > phi_3_x_r1[0]
# 	xu3 = stl.Predicate("xu") < phi_3_x_r1[1]

# 	yl3 = stl.Predicate("yl") > phi_3_y_r1[0]
# 	yu3 = stl.Predicate("yu") < phi_3_y_r1[1]

# 	zl3 = stl.Predicate("zl") >phi_3_z_r1[0]
# 	zu3 = stl.Predicate("zu") < phi_3_z_r1[1]
# 	phi_3 = stl.Eventually(stl.Always((xl3&xu3)&(yl3&yu3)&(zl3&zu3)))


# 	phi_1_2 = stl.Until(phi_1,phi_2)
# 	phi_1_2_3 = stl.Until(phi_1_2,phi_3)


# 	xl4 = stl.Predicate("xl") > psi_x[0]
# 	xu4 = stl.Predicate("xu") < psi_x[1]

# 	yl4 = stl.Predicate("yl") > psi_y[0]
# 	yu4 = stl.Predicate("yu") < psi_y[1]

# 	zl4 = stl.Predicate("zl") > psi_z[0]
# 	zu4 = stl.Predicate("zu") < psi_z[1]
# 	psi = stl.Always(stl.Not((xl4&xu4)&(yl4&yu4)&(zl4&zu4)))


# 	phi = psi & phi_1_2_3

# 	@staticmethod
# 	def evaluate(trajectory: torch.Tensor, step_number: int, agent) -> float:
# 		with torch.no_grad():

# 			x_traj = trajectory[:, 0].tolist()
# 			y_traj = trajectory[:, 1].tolist()
# 			z_traj = trajectory[:, 2].tolist()
# 			t = np.arange(step_number, step_number + trajectory.shape[0]).tolist()
# 			trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
# 			rob = stl.compute_robustness(MCTS_STL_spec_R1.phi, trace, synchronized=True)
# 			return rob


class MCTS_STL_spec_other:

	def evaluate(trajectory: torch.Tensor, step_number: int, goal_position_raw: torch.Tensor,agent) -> float:
		# with torch.no_grad():
		# 	# print(int(goal_position_raw[0,0])-0.02)
		# 	xl = stl.Predicate("xl") > int(goal_position_raw[0,0])-0.02
		# 	xu = stl.Predicate("xu") < int(goal_position_raw[0,0])+0.02

		# 	yl = stl.Predicate("yl") > int(goal_position_raw[0,1])-0.02
		# 	yu = stl.Predicate("yu") < int(goal_position_raw[0,1])+0.02

		# 	zl = stl.Predicate("zl") > int(goal_position_raw[0,2])-0.02
		# 	zu = stl.Predicate("zu") < int(goal_position_raw[0,2])+0.02
		# 	phi = stl.Eventually(stl.Always((xl&xu)&(yl&yu)&(zl&zu)))
		# 	x_traj = trajectory[:, 0].tolist()
		# 	y_traj = trajectory[:, 1].tolist()
		# 	z_traj = trajectory[:, 2].tolist()
		# 	t = np.arange(step_number, step_number + trajectory.shape[0]).tolist()
		# 	trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
		# 	rob = stl.compute_robustness(phi, trace, synchronized=True)
			rob = 0
			return rob

####################################
# check1 = downwind_STL_spec()
  

# Eventually Globally
# print (MCTS_STL_spec.evaluate(pred_traj_1,0,goal_position, 0))
# print (EfficientSTLSpec.evaluate(pred_traj_2,0,goal_position, 0))


# class base_STL_spec:
# #     def __init__(self):

#     @staticmethod
#     def evaluate(trajectory: torch.Tensor, step_number: int, goal_position, agent) -> float:
#         with torch.no_grad():




#             x_traj = trajectory[:, 0].tolist()
#             y_traj = trajectory[:, 1].tolist()
#             z_traj = trajectory[:, 2].tolist()
#             t = np.arange(step_number, step_number + trajectory.shape[0]).tolist()
#             trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
#             rob = stl.compute_robustness(phi_2, trace, synchronized=True)
#             return rob


   
# check2 = base_STL_spec()
# class final_STL_spec:
# #     def __init__(self):

#     @staticmethod
#     def evaluate(trajectory: torch.Tensor, step_number: int, goal_position, agent) -> float:
#         with torch.no_grad():

#             phi_3_x = [1.7, 3.0]
#             phi_3_y = [-1.8, 0.02]
#             phi_3_z = [0.3, 0.5]

#             xl = stl.Predicate("xl") > phi_3_x[0]
#             xu = stl.Predicate("xu") < phi_3_x[1]

#             yl = stl.Predicate("yl") > phi_3_y[0]
#             yu = stl.Predicate("yu") < phi_3_y[1]

#             zl = stl.Predicate("zl") >phi_3_z[0]
#             zu = stl.Predicate("zu") < phi_3_z[0]
#             phi_3 = stl.Eventually(stl.Always((xl&xu)&(yl&yu)&(zl&zu)))
#             x_traj = trajectory[:, 0,0].tolist()
#             y_traj = trajectory[:, 0,1].tolist()
#             z_traj = trajectory[:, 0,2].tolist()
#             t = np.arange(step_number, step_number + trajectory.shape[0]).tolist()
#             trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
#             rob = stl.compute_robustness(phi_3, trace, synchronized=True)
#             return rob
			
# check3 = final_STL_spec()
# class overfly_STL_spec:
# #     def __init__(self):

#     @staticmethod
#     def evaluate(trajectory: torch.Tensor, step_number: int, goal_position, agent) -> float:
#         with torch.no_grad():

#             psi_x = [0.0, 1.45]
#             psi_y = [-0.01, 0.01]
#             psi_z = [0.1, 0.6]

#             xl = stl.Predicate("xl") > psi_x[0]
#             xu = stl.Predicate("xu") < psi_x[1]

#             yl = stl.Predicate("yl") > psi_y[0]
#             yu = stl.Predicate("yu") < psi_y[1]

#             zl = stl.Predicate("zl") >psi_z[0]
#             zu = stl.Predicate("zu") < psi_z[0]
#             phi_3 = stl.Eventually(stl.Always((xl&xu)&(yl&yu)&(zl&zu)))
#             x_traj = trajectory[:, 0,0].tolist()
#             y_traj = trajectory[:, 0,1].tolist()
#             z_traj = trajectory[:, 0,2].tolist()
#             t = np.arange(step_number, step_number + trajectory.shape[0]).tolist()
#             trace = dict(xl=stl.Signal(x_traj, t), xu=stl.Signal(x_traj, t),yl=stl.Signal(y_traj, t), yu=stl.Signal(y_traj, t),zl=stl.Signal(z_traj, t), zu=stl.Signal(z_traj, t))
#             rob = stl.compute_robustness(psi_3, trace, synchronized=True)
#             return rob
# # stl.Until()
# stl.Until(stl.Eventually(stl.Always((xld&xud)&(yld&yud)&(zld&zud))),stl.Eventually(stl.Always((xlb&xub)&(ylb&yub)&(zlb&zub))))