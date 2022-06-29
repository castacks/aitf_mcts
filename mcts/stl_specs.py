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
phi_2_y = [-2.5, -1.0] 
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
	trajs_y = ways[:,1].tolist()
	trajs_z = ways[:,2].tolist()
	r2 = np.array([[1.48,0.0,0.38]])
	diff_trajs = np.diff(ways, axis = 0)
	# print (diff_trajs,diff_trajs[:,1])
	diff_angle = np.arctan2(diff_trajs[:,1],diff_trajs[:,0])
	# print (diff_angle*180/np.pi)
	# print (diff_angle.shape)

	diff_angle_full = np.concatenate((diff_angle,diff_angle[-1:]))
	# print (diff_angle_full*180/np.pi)

	diff_angle_full = np.where(diff_angle_full < -1.57, diff_angle_full + 2 * np.pi, diff_angle_full)

	# diff_angle_full = diff_angle_full.tolist()
	phi_1u = 0.349066 - diff_angle_full
	phi_1l = diff_angle_full + 0.349066
	phi_2u = 2.00713 - diff_angle_full
	phi_2l = diff_angle_full - 1.13446
	phi_3u = 3.40339 - diff_angle_full
	phi_3l = diff_angle_full - 2.87979

	phi_1u = phi_1u.tolist()
	phi_1l = phi_1l.tolist()
	phi_2u = phi_2u.tolist()
	phi_2l = phi_2l.tolist()
	phi_3u = phi_3u.tolist()
	phi_3l = phi_3l.tolist()

	diff_trajs = np.concatenate((diff_trajs,diff_trajs[-1:]))
	# print (diff_angle_full*180/np.pi)
	# print('phi_3l',phi_3l)

	vels_x = diff_trajs[:,0].tolist()
	vels_y = diff_trajs[:,1].tolist()
	vels_z = diff_trajs[:,2].tolist()

	# print(len(vels_x))

	# diff_from_runway = ways - r2
	# print(diff_from_runway)
	# print("ways",ways.shape,r2.shape)
	if torch.is_tensor(ways):
		ways = ways.numpy()
	# sqrA = torch.sum(torch.pow(ways, 2), 1, keepdim=True).expand(ways.shape[0], r2.shape[0])
	# sqrB = torch.sum(torch.pow(r2, 2), 1, keepdim=True).expand(r2.shape[0], ways.shape[0]).t()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], r2.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(r2, 2), 1).reshape(r2.shape[0], 1), (r2.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_runway = np.sqrt(sqrA - 2*np.matmul(ways, r2.transpose()) + sqrB)
	#
	# print(L2_dist_runway.shape)
	L2_dist_runway = L2_dist_runway.squeeze().tolist()
	# print(L2_dist_runway)

	# print("trajs_x",trajs_x)
	xl_r1 = np.array(trajs_x) - 0.5
	xu_r1 = 2.5 - np.array(trajs_x)
	yl_r1 = np.array(trajs_y) + 2.5
	yu_r1 = -1.0 - np.array(trajs_y)
	zl_r1 = np.array(trajs_z) - 0.5
	zu_r1 = 0.7 - np.array(trajs_z)

	xl_r2 = np.array(trajs_x) - 2.5
	xu_r2 = 4.0 - np.array(trajs_x)
	yl_r2 = np.array(trajs_y) + 2.5
	yu_r2 = -0.08 - np.array(trajs_y)
	zl_r2 = np.array(trajs_z) - 0.3
	zu_r2 = 0.5 - np.array(trajs_z)
	# print("sub",xl_r1)
	# user_details = [{'name' : x, 'rank' : trajs_x.index(x)} for x in trajs_x]
	xl_r1 = list(enumerate(xl_r1))
	xu_r1 = list(enumerate(xu_r1))
	yl_r1 = list(enumerate(yl_r1))
	yu_r1 = list(enumerate(yu_r1))
	# zl_r1 = list(enumerate(zl_r1))
	# zu_r1 = list(enumerate(zu_r1))
	xl_r2 = list(enumerate(xl_r2))
	xu_r2 = list(enumerate(xu_r2))
	yl_r2 = list(enumerate(yl_r2))
	yu_r2 = list(enumerate(yu_r2))
	trajs_x = list(enumerate(trajs_x))
	trajs_y = list(enumerate(trajs_y))
	trajs_z = list(enumerate(trajs_z))

	phi_1u = list(enumerate(phi_1u))
	phi_1l = list(enumerate(phi_1l))
	phi_2u = list(enumerate(phi_2u))
	phi_2l = list(enumerate(phi_2l))
	phi_3u = list(enumerate(phi_3u))
	phi_3l = list(enumerate(phi_3l))
	# print('phi_3l',phi_3l)


	vel_x = list(enumerate(vels_x))
	vel_y = list(enumerate(vels_y))
	vel_z = list(enumerate(vels_z))
	# print('vel_z',vel_z)

	L2_runway = list(enumerate(L2_dist_runway))

	# print('trajs_z',trajs_z)
	# # stl
	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	# spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
	spec.name = 'runway r2'
	spec.declare_var('iq', 'float')
	spec.declare_var('up', 'float')
	spec.declare_var('x', 'float')
	spec.declare_var('xl_r1', 'float')
	spec.declare_var('xu_r1', 'float')
	spec.declare_var('yl_r1', 'float')
	spec.declare_var('yu_r1', 'float')
	spec.declare_var('zl_r1', 'float')
	spec.declare_var('zu_r1', 'float')

	spec.declare_var('xl_r2', 'float')
	spec.declare_var('xu_r2', 'float')
	spec.declare_var('yl_r2', 'float')
	spec.declare_var('yu_r2', 'float')
	spec.declare_var('y', 'float')
	spec.declare_var('yl', 'float')
	spec.declare_var('yu', 'float')
	spec.declare_var('z', 'float')

	spec.declare_var('zl', 'float')
	spec.declare_var('zu', 'float')
	spec.declare_var('final', 'float')

	spec.declare_var('out', 'float')

	spec.declare_var('phi_1u_head', 'float')
	spec.declare_var('phi_1l_head', 'float')

	spec.declare_var('phi_2u_head', 'float')
	spec.declare_var('phi_2l_head', 'float')

	spec.declare_var('phi_3u_head', 'float')
	spec.declare_var('phi_3l_head', 'float')

	# spec.set_var_io_type('xl', 'input')
	# spec.set_var_io_type('xu', 'input')

	spec.set_var_io_type('yl', 'input')
	spec.set_var_io_type('yu', 'input')

	spec.set_var_io_type('zl', 'input')
	spec.set_var_io_type('zu', 'input')

	spec.declare_var('vel_x', 'float')
	spec.declare_var('vel_y', 'float')
	spec.declare_var('vel_z', 'float')
	spec.declare_const('threshold_1', 'float', '3')

	spec.declare_const('threshold_2', 'float', '5')
	# spec.declare_var('x', 'float')
	# spec.declare_var('iq', 'float')
	# spec.declare_var('up', 'float')
	# spec.declare_var('xl', 'float')
	# spec.declare_var('xu', 'float')
	# spec.declare_var('y', 'float')
	# spec.declare_var('yl', 'float')
	# spec.declare_var('yu', 'float')
	# spec.declare_var('z', 'float')
	# spec.declare_var('zl', 'float')
	# spec.declare_var('zu', 'float')
	# spec.declare_var('out', 'float')
	# spec.declare_var('out', 'float')
	# spec.set_var_io_type('xl', 'input')
	# spec.set_var_io_type('xu', 'input')

	# spec.set_var_io_type('yl', 'input')
	# spec.set_var_io_type('yu', 'input')

	# spec.set_var_io_type('zl', 'input')
	# spec.set_var_io_type('zu', 'input')

	# spec.declare_var('phi_1', 'float')
	# spec.declare_var('phi_2', 'float')
	# spec.declare_var('phi_3', 'float')
	# # spec.set_var_io_type('out', 'output')
	# # spec.spec = '(eventually[0:0] ((always[0:0] (xl > 0.5)) and (always[0:0] (xu < 3.0))) )'
	# spec.add_sub_spec = ('phi_1 = (eventually[0:0] (always[0:0]( (xl > 0.5) and (xu < 2.5) ) and ( (yl > -2.5) and (yu < -1.0) ) and ( (zl > 0.5) and (zu < 0.7) ) ) )')
	# spec.add_sub_spec = ('phi_2 = (eventually[0:0] (always[0:0]( (xl > 2.5) and (xu < 4.0) ) and ( (yl > -2.5) and (yu < -0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')
	# spec.add_sub_spec = ('phi_3 = (eventually[0:0] (always[0:0]( (xl > 1.5) and (xu < 4.0) ) and ( (yl > -0.08) and (yu < 0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')

	# spec.spec = '( ((phi_1) until[0:0] (phi_2)) until[0:0] (phi_3) )'

	# spec.spec = 'eventually(( ( (xl > 0.5) and (xu < 2.5) ) and ( (yl > -2.5) and (yu < -1.0) )  ) and eventually( ( (xl > 2.5) and (xu < 4.0) ) and ( (yl > -2.5) and (yu < -0.08) ) ) )'
	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08)) ))'

	# spec.spec =eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08)) )) 
	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -1.0)) and eventually( ((x > 1.5) and (x < 4.0) ) and ((y > -0.5) and (y < 0.2)) ) ))'
	
	
	spec.spec = 'eventually( (( (x >-2.5) and (x < 0.5)) and ( (y > -2) and (y < -1) )  and ( (z > 0.72) and (z < 0.96) )  ) and eventually( ((x > 1.5) and (x < 3.0) ) and ((y > -2) and (y < 0.2))  and eventually(((x > 1.3) and (x < 1.5) ) and ((y > -0.2) and (y < 0.2)))))'

	# spec.spec = 'eventually((((x > 1.3) and (x < 1.5) ) and ((y > -0.2) and (y < 0.2)) and (z < 0.4)))'
	# spec.spec = '(((x > 1.5) and (x < 4.0) ) and ((y > -0.5) and (y < 0.2))) until (((x >-1.5) and (x < 0.5)) and ( (y > -3) and (y < -2) ))'   
	# spec.spec =  '(((x >-1.5) and (x < 0.5)) and ( (y > -3) and (y < -2) )) and not((((x > 1.5) and (x < 4.0) ) and ((y > -0.5) and (y < 0.2))))  until (((x > 1.5) and (x < 4.0) ) and ((y > -0.5) and (y < 0.2))) '   

	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  and ((z > 0.3) and (z < 0.5)) ))'
	# spec.spec = '(eventually(always((( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) ))))) until 	(eventually(always(( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  and ((z > 0.3) and (z < 0.5)) ))))'
	# spec.spec = '(eventually(always((( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) )))) until 	(eventually(always(( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  ))))'

	# spec.add_sub_spec('iq = x>0.5')
	# spec.add_sub_spec('up = x<2.5')
	# spec.spec = 'out = ((iq) until 	 (up) )'

	# spec.spec = 'eventually(( ( (x > 0.5) and (x < 2.5) ) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) ) ) and eventually( ( (x > 2.5) and (x < 4.0) ) and ( (y > -2.5) and (y < -0.08) ) and ( (z > 0.3) and (z < 0.5) ) ) )'
	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	# out = spec.evaluate(['x', trajs_x],['y', trajs_y])
	out = spec.evaluate(['xl_r1', xl_r1],['xu_r1', xu_r1],['yl_r1', yl_r1],['yu_r1', yu_r1],['zl_r1', zl_r1],['zu_r1', zu_r1],['xl_r2', xl_r2],['xu_r2', xu_r2],['yl_r2', yl_r2],['yu_r2', yu_r2],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z],['x',trajs_x],['y',trajs_y],['z',trajs_z],['vel_x',vel_x],['vel_y',vel_y],['vel_z',vel_z],['final',L2_runway],['phi_1l_head',phi_1l],['phi_1u_head',phi_1u],['phi_2l_head',phi_2l],['phi_2u_head',phi_2u],['phi_3l_head',phi_3l],['phi_3u_head',phi_3u ])

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