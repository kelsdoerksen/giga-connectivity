"""
Sentinel-2 imagery extarction for school location points
"""

from sentinelsat import SentinelAPI
from shapely.geometry import Polygon
import numpy as np
from scipy.interpolate import RectBivariateSpline


def get_tallinn_polygon(swap_coordinates=False):
    tln_points = [(25.2350806, -20.2066273), (25.3152243, -24.9947434), (25.6978741, -24.9859972), (25.8626712, -24.622995), (25.2168433, -25.7576782), (27.2186993, -20.1328199), (25.4556447, -24.6748963), (25.2350806, -20.2066273), (25.3152243, -24.9947434), (25.9136956, -24.6538453), (25.9136956, -24.6538453), (25.2350806, -20.2066273), (25.2168433, -25.7576782), (25.6978741, -24.9859972), (27.2186993, -20.1328199), (25.847013, -24.6447436), (25.560257, -24.6543768), (25.4556447, -24.6748963), (25.847013, -24.6447436), (25.8626712, -24.622995), (25.373073, -24.5547355), (25.364799, -24.4147381), (25.9544492, -24.6770331), (28.4178865, -21.9732006), (21.9287096, -26.6624954), (20.6877527, -26.8948576), (25.9157234, -24.6718514), (25.5075556, -24.3882364), (25.4923482, -25.6198242), (22.7488737, -20.5486285), (27.6909756, -20.6835432), (25.9388931, -24.6669355), (25.506390300000003, -24.400103100000003), (27.2502913, -22.1754), (21.4382753, -18.7096469), (27.2593993, -20.221422), (25.561364, -25.3453187), (25.6309867, -25.369884), (23.8871255, -23.9776625), (21.7672763, -26.7136142), (27.392574600000003, -20.5610513), (21.641446, -21.7029027), (25.499143300000004, -25.3874522), (28.7658234, -21.8166259), (25.542373, -24.675065), (25.4405959, -25.4756456), (27.4239327, -22.7277115), (27.1957523, -20.510964), (27.2267603, -20.863318), (22.1608243, -19.3876839), (25.6796622, -25.232033800000004), (26.1391837, -24.4181616), (27.707398699999995, -20.571005300000003), (27.6802344, -20.9540029), (24.5012343, -18.1512069), (27.29161, -20.6333525), (26.4824904, -22.8666261), (20.8642167, -22.1538775), (20.3172473, -22.192081), (22.7866803, -23.6684061), (20.6169677, -22.0916005), (25.1492822, -17.8000307), (24.5852467, -18.0751924), (25.2332828, -17.7957137), (25.5211707, -24.400511), (26.138243300000003, -24.418565), (27.5622963, -22.384409), (26.469515500000004, -23.0396266), (23.5051853, -24.7004231), (26.707145, -22.3962159), (24.7481761, -23.884328), (21.8380243, -26.6583942), (24.5118263, -20.473418), (20.1928903, -22.992894), (21.9940283, -26.6438322), (25.6977254, -24.6644622), (26.1565014, -24.4007786), (27.2231608, -22.6646603), (21.8323153, -23.914387100000003), (29.0366107, -22.2067755), (25.8517472, -24.2583176), (25.8355104, -23.3599082), (25.2028639, -25.75732), (27.7579043, -22.786835), (25.9183712, -24.6641556), (27.5368708, -22.5195308), (26.4510287, -23.8404895), (25.029927800000003, -24.1022323), (26.4378743, -24.6045811), (27.4123647, -20.6680668), (23.4276448, -19.982701), (25.8984849, -24.4752459), (26.145049, -24.3918072), (21.7800063, -24.1138191), (27.2959413, -22.856463), (26.4317123, -24.6018421), (24.6460543, -17.9865549), (24.5620606, -25.7679479), (23.5799109, -24.3935527), (27.1336393, -20.299987), (27.4724405, -21.4495258), (28.4139273, -21.971823), (25.5436823, -24.6651011), (27.0653757, -20.4991725), (25.291599, -25.4431466), (26.8341977, -23.1020562), (25.3341227, -24.9761326), (25.3152243, -24.9945837), (23.8788747, -20.3265983), (27.2745782, -20.8999794), (22.9776933, -25.335716100000003), (20.1184173, -22.389493), (27.1230473, -20.676647), (27.327251300000004, -20.797220000000003), (27.4775123, -22.718381), (25.5929949, -24.7726636), (27.0551423, -20.514598), (27.5887404, -20.5365266), (25.404452300000003, -25.6574941), (27.4582683, -21.44227), (27.2655287, -20.7984745), (27.4441145, -20.62082), (27.644548800000003, -22.5843706), (26.0371963, -24.540967), (27.338337300000003, -20.99652), (27.5659823, -23.112276100000003), (27.3113233, -22.718298), (27.8830673, -21.297264), (26.6769063, -20.835264), (27.6250751, -22.5987112), (27.7482053, -21.864873), (25.2705668, -25.3129467), (25.6582973, -21.149615), (25.644935, -24.7247933), (27.2767933, -23.2973941), (26.5760132, -23.0444023), (26.1223733, -24.6257212), (25.2319352, -24.9061798), (27.6123163, -22.681061), (27.8247963, -22.323602), (27.5912113, -22.375597), (26.561649300000003, -22.4337378), (26.9629627, -23.139833), (26.4069183, -22.634903), (25.352363, -24.9830774), (25.8688599, -24.8752264), (24.9481573, -21.262027), (28.5958113, -22.063689), (25.552938600000004, -25.2010315), (27.631166, -22.7523567), (26.6589497, -23.6844725), (23.4197456, -19.9934843), (27.6500964, -20.5332737), (25.4071981, -24.7781213), (21.7655543, -23.9912261), (27.5162533, -20.622181), (26.36178, -22.9372098), (26.019990300000003, -21.210857), (24.1298563, -20.217667), (25.3457387, -24.990015), (27.1163663, -20.559697), (26.158696, -20.2087389), (20.2843723, -23.1351841), (25.5282637, -24.409322600000003), (20.6532443, -22.42962), (26.741622, -22.423015300000003), (27.1331519, -20.3005557), (27.5171233, -20.515492), (22.1914703, -19.6618909), (27.1846157, -20.7899025), (25.5875989, -24.960337300000003), (27.477429600000004, -21.2005985), (26.8619413, -24.1879321), (22.566512300000003, -25.7605232), (26.0353483, -24.5720391), (25.7474947, -25.0388502), (26.0755873, -24.438344), (27.4997986, -21.1743826), (26.7888145, -22.2731276), (26.6933067, -23.4037058), (27.1312995, -22.5513107), (25.6590823, -18.5346059), (24.3126354, -18.0393722), (27.7489353, -21.87196), (25.0944526, -25.7388494), (25.3382483, -24.9716401), (26.0099963, -24.4283271), (25.5409035, -25.634477800000003), (26.0608487, -24.4366526), (27.4147232, -22.8669018), (25.4629826, -24.914794), (26.4730997, -24.545910600000003), (20.816321, -26.8223907), (26.080615, -24.3738499), (27.573336, -22.722667), (27.453745, -21.4287119), (26.7120894, -22.4036831), (24.4045543, -18.0133549), (27.533953000000004, -21.1876764), (25.5112405, -24.4105407), (27.1397274, -22.5373979), (27.2146327, -20.8215665), (27.3568257, -20.7537325), (27.5276613, -23.0237121), (27.944927000000003, -22.1917747), (26.1577217, -24.3751826), (25.3367523, -24.9870441), (22.6993573, -20.480582), (27.3207755, -20.5385444), (27.5096453, -22.887802), (27.0009993, -20.486651), (28.8283267, -21.8784025), (27.1287, -20.3228409), (27.6874718, -20.766319600000003), (27.689044300000003, -22.655632), (22.1771853, -18.7342309), (22.4136883, -18.8159839), (26.7061287, -22.3894337), (27.2983143, -21.919123000000003), (21.8428403, -18.3896039), (23.6686257, -19.7654865), (26.5200218, -23.030260100000003), (26.4035223, -24.6299501), (26.7030903, -22.407336), (27.6733385, -20.8587562), (25.8274547, -24.8483496), (25.8698094, -24.8679091), (25.8696591, -24.7144836), (25.4964797, -25.6540956), (26.8284685, -23.091447), (27.7529493, -21.865593), (25.6794772, -25.228311), (27.3992067, -22.4104125), (26.8216503, -23.110207100000004), (27.469701300000004, -21.283731), (27.4944475, -21.1659135), (26.4288763, -22.491001), (23.4216231, -19.9927735), (27.7434045, -21.8807249), (27.859606300000003, -21.984008), (25.6311012, -25.5086184), (28.0444877, -21.9030315), (27.4615253, -21.4413843), (24.566567300000003, -21.070384), (22.3972063, -26.0252972), (27.6248949, -20.836212300000003), (23.411423000000003, -19.9916851), (28.3985043, -22.412914), (21.8675433, -24.025653100000003), (26.6923044, -22.3926474), (27.5859613, -20.720003), (24.6855648, -25.8177775), (26.9225803, -22.114924), (24.4013613, -21.031948), (27.0353387, -20.4903425), (23.2818827, -25.2748346), (27.4376463, -20.5041906), (26.3146353, -24.017472100000003), (24.9591177, -24.1674815), (22.32905, -20.3824985), (25.8222055, -23.6022742), (25.9653107, -24.6685526), (27.227987899999995, -20.3249086), (25.4896417, -24.4127896), (24.5298707, -24.0932485), (25.675134, -24.2363018), (25.048271300000003, -24.522617), (24.5886564, -23.7350516), (26.140537100000003, -24.3931262), (26.7023553, -22.38212), (25.5239761, -24.3823294), (24.3079591, -24.0029389), (26.1677813, -24.2925751), (25.9392087, -24.5787346), (25.8596532, -24.8727297), (25.3098077, -25.5374027), (25.860722, -24.8730972), (26.7174771, -22.362929), (27.4451043, -22.570692), (26.8243643, -23.6469171), (26.4411069, -21.8671072), (24.8576697, -21.2021565), (24.2509353, -20.240138), (27.375551300000005, -22.947214), (26.9996697, -22.8026245), (26.831885600000003, -23.1338238), (25.6786786, -25.2191831), (25.6275831, -25.0911146), (27.4723433, -23.1875476), (26.7138043, -22.41309), (25.8751598, -24.5686508), (22.292448, -19.1131438), (25.9083523, -24.309268), (25.5005611, -25.4336431), (22.6884997, -25.7075852), (25.3285597, -24.649585), (25.627517, -24.9095358), (25.8641984, -24.0666288), (25.4770773, -25.2923515), (26.6072982, -20.6440052), (21.1778337, -19.5313425), (27.1125208, -22.5514872), (25.7446788, -23.4309698), (26.8052163, -23.1080451), (25.7681719, -24.602951), (25.3385843, -25.7504852), (23.0649647, -23.5687735), (27.8344577, -21.9688875), (23.4243805, -19.9844554), (22.3269673, -20.62771), (27.495405100000003, -21.1483336), (25.3633513, -23.86203), (26.9251787, -23.8516555), (26.819864000000003, -23.1158909), (21.872414300000003, -22.820781), (25.88123, -24.6619416), (27.869807300000005, -21.978556), (26.2179713, -24.59501), (25.5898263, -24.135781100000003), (22.2049513, -22.854826), (25.4289991, -24.7689107), (26.074862300000003, -23.6647661), (26.1814783, -23.7355831), (23.1140783, -24.345723100000004), (25.4607794, -25.0907092), (25.6059819, -25.4726162), (25.2724923, -24.1831991), (24.9582853, -25.431291100000003), (24.7980038, -25.8126426), (26.1442225, -24.3688844), (26.1170783, -24.401473100000004), (27.8371886, -21.9863758), (27.229251300000005, -22.793353), (26.4635077, -24.5787876), (27.5555293, -20.801208), (20.7084253, -26.8676582), (25.9138353, -24.6619797), (23.4119457, -19.998611), (27.040930300000003, -23.522228100000003), (23.7566793, -24.597419), (26.1855698, -23.1233697), (27.1395179, -23.414063100000003), (28.428626300000005, -21.97624), (24.6273726, -24.903077), (27.1887536, -22.8042279), (21.9918823, -18.5795999), (23.2809303, -25.274312100000003), (25.1131637, -24.9556816), (27.134933, -23.1091127), (25.868951100000004, -24.8907139), (27.4530538, -21.4058407), (22.9551263, -20.362647), (21.0797133, -19.8836699), (26.1701532, -24.3814186), (27.8681583, -21.956092), (25.4923323, -24.3983191), (25.913375, -24.6323564), (25.690159, -25.2049259), (27.345737100000004, -21.0767065), (25.9075758, -24.6765057), (26.1486273, -24.41076), (22.4011093, -21.04634), (25.9696593, -24.654966100000003), (27.0700127, -20.8954373), (21.8821883, -26.6666002), (25.6698912, -24.8556582), (25.2358945, -25.5586295), (25.4935517, -25.5438656), (25.5470454, -25.5521192), (27.226868300000003, -20.863377), (27.1263997, -22.5282375), (25.3591227, -24.972594600000004), (24.8985988, -24.8798232), (25.658265300000004, -23.4781331), (26.702249, -22.3820817), (25.8643859, -24.6131112), (25.9252505, -24.6682467), (25.8958214, -24.6905555), (24.8267833, -24.8631331), (25.6159153, -24.544858), (27.4900249, -21.1701528), (22.6035643, -20.631527), (20.0650643, -22.274953), (27.1100094, -22.5579159), (25.5388378, -24.395518300000003), (26.5870591, -23.2643569), (23.4328562, -20.0029235), (25.9059583, -24.5612599), (25.626596300000003, -24.9067636), (27.7299194, -21.883459100000003), (23.4982823, -19.9416349), (25.3872433, -25.4647283), (25.8874422, -24.4729219), (25.5099213, -25.152573), (25.1362707, -25.5420026), (27.650051300000005, -20.588958), (25.9018404, -24.679471300000003), (23.8858483, -20.319302), (25.3468369, -24.9900393), (26.0060783, -24.1393216), (24.9024743, -25.7935092), (21.6485063, -21.703303), (27.7554762, -22.7937906), (25.863313100000003, -24.8591578), (25.357660600000003, -24.9975281), (25.3729811, -25.5476723), (27.5334978, -23.0354027), (25.5432993, -24.258612), (25.5766729, -25.5654326), (25.6798789, -25.2101314), (26.0775767, -24.4061216), (25.3094959, -24.813544600000004), (25.1352782, -25.6413379), (25.1779783, -25.195409100000003), (23.4114334, -19.9850805), (23.9222789, -24.5160064), (21.7704603, -22.102198), (22.7401293, -18.7219919), (25.9260326, -24.6310117), (28.4092273, -21.961075), (27.821148699999995, -21.9866075), (22.2905355, -19.1139429), (25.787149300000003, -24.470647), (22.5761663, -18.8401309), (22.388077300000003, -19.863915), (22.2819403, -18.8656049), (21.8350698, -18.277116), (26.399249600000005, -24.6325568), (25.2217253, -17.901274899999997), (27.5177894, -21.1787983), (26.7207562, -22.4405484), (24.4782303, -21.090594), (26.7434198, -22.4253698), (27.49731210000001, -21.1416518), (25.9192951, -24.6400305), (27.8467744, -21.9864179), (25.90868, -24.6665793), (26.8117287, -23.1264906), (21.884597, -18.3172984), (26.4944613, -20.580297), (24.8741698, -25.1068466), (25.9324523, -24.6118724), (25.1353867, -25.389913600000003), (24.6023383, -21.170503), (27.6498363, -20.588971), (23.2240313, -20.200612), (25.9421459, -24.6082153), (24.4792262, -25.3237654), (26.7905703, -22.020755), (27.331530600000004, -22.6126296), (25.243011300000003, -23.98127), (26.2464682, -20.1133617), (27.608162300000004, -20.582424), (27.249761300000003, -20.85382), (26.7898713, -22.7679), (26.4255723, -21.230077), (25.2280707, -24.323355600000003), (25.9355609, -24.636022), (27.4507423, -21.368892), (26.26801, -23.5993055), (22.4472077, -20.3114022), (27.0807883, -20.937811), (26.4876294, -19.862381), (24.8016189, -24.6599755), (25.0057173, -24.8890151), (27.3814004, -21.1898337), (25.9248959, -24.6213431), (26.9958109, -20.5004646), (25.8874404, -24.6663186), (25.9448479, -24.621606), (23.67348, -23.8668721), (25.9322306, -24.6190663), (25.4831202, -20.0426916), (27.5232668, -22.8891907), (24.6027133, -21.17045), (25.4032231, -24.7686331), (25.7480961, -25.034025600000003), (27.6373853, -22.651659), (27.3317143, -21.099218), (23.4047189, -19.9754135), (21.8518483, -22.163427), (27.3911313, -22.724051), (24.7237054, -24.6021107), (22.4171737, -22.1186545), (26.8778779, -21.3065362), (25.4352537, -25.5672426), (26.8956113, -22.78297), (24.5963023, -24.7680901), (25.3651313, -25.384433), (25.5058603, -24.4129931), (21.879886300000003, -26.666304200000003), (25.7988126, -24.6560376), (27.2783023, -21.786547), (23.7120641, -25.4524131), (22.401943300000003, -26.0198182), (25.5349243, -25.411422100000003), (25.6631703, -24.478205), (28.2362563, -22.507602), (25.5237675, -24.6520628), (26.4598389, -23.7466256), (23.2991669, -24.0613728), (21.9578273, -18.351868), (23.871270300000003, -19.4284779), (22.2814107, -19.357727800000003), (25.8282833, -24.644341600000004), (25.0271177, -24.085517000000003), (27.4444023, -21.437092), (25.8813725, -24.6531541), (25.4731017, -24.4015276), (25.8352946, -24.2547693), (22.6832053, -25.7952632), (26.4951453, -23.045678), (26.801523, -23.0869614), (25.0051609, -24.889263600000003), (23.3943613, -20.0145159), (27.5343837, -21.4626489), (27.099552600000003, -22.5418645), (27.4576723, -20.570354), (26.6880923, -22.4025016), (23.4507794, -19.9605759), (27.6588869, -21.119364800000003), (28.4284113, -21.968887), (25.0370426, -24.6905983), (25.3119719, -25.5936376), (24.4109923, -21.037518), (21.6016913, -23.4119601), (23.2188323, -24.9204641), (25.401319, -24.5053314), (27.8124503, -21.986962), (25.69424, -25.1989054), (21.220654, -23.4432624), (22.9705353, -23.9507551), (25.4897661, -24.0245151), (27.5357203, -22.357896), (24.7321597, -24.600496300000003), (25.890795, -24.6570948), (25.5108227, -25.152594600000004), (21.2131083, -21.360545), (25.403899, -24.8039204), (23.8577874, -24.5648903), (25.4449652, -21.9468979), (24.2217207, -24.4154715), (27.4134203, -21.383479), (28.751344300000003, -22.267492), (25.3128923, -24.959572100000003), (20.8259843, -23.1508301), (24.5355064, -20.195325600000004), (27.5260923, -21.202289), (25.5794583, -21.43374), (26.0174918, -23.5405203), (27.5163155, -21.2044587), (26.7251773, -22.405887), (27.337380300000003, -21.013459), (27.5167672, -21.187774800000003), (27.5164341, -21.1958174), (25.5776343, -25.5065001), (25.7413172, -25.0378071), (26.4785092, -20.5872885), (24.2291203, -24.914978100000003), (25.794227300000003, -20.154887), (26.8934989, -20.7504407), (24.8768733, -21.201181), (23.4052107, -19.9922819), (28.8747326, -22.1750266), (27.4510603, -21.450985), (27.500357300000005, -21.129948), (26.2186907, -20.5627699), (24.7306093, -24.6063223), (27.5266528, -21.1845918), (26.413239500000003, -23.1326149), (22.9714923, -23.950401100000004), (27.7695517, -21.8791275), (21.3659835, -26.2827328), (21.750255300000003, -23.9842791), (26.8326923, -22.214366), (25.9387113, -21.815399), (25.8623663, -24.513229), (27.1515213, -23.10699), (25.3684258, -24.875122), (25.583677, -21.4096214), (25.8522999, -24.8548307), (27.5596043, -23.032004), (20.4969879, -23.5588236), (28.3874853, -22.404445000000003), (21.2070213, -22.154975), (25.900076, -24.643283800000003), (23.6319673, -24.708919100000003), (25.8897127, -24.6459078), (25.8457397, -24.615286800000003), (27.2785617, -21.7865415), (25.908299300000003, -24.3093281), (25.306330300000003, -23.6728), (26.8484401, -23.1040885), (27.519257, -21.1643644), (22.1716683, -19.376552), (25.2752913, -24.47736), (25.5718259, -24.3979725), (27.271282300000003, -22.04925), (26.3910373, -22.580558), (25.667361, -24.557565), (25.8639009, -24.6665), (23.6550338, -20.150329), (25.9058048, -24.6251365), (26.389806300000004, -22.580949), (26.7183517, -22.3857375), (27.746948, -23.0060917), (25.4631109, -24.406921), (27.1205214, -22.5349896), (27.225299300000003, -21.098425), (27.196175300000004, -20.1048889), (25.44017, -25.0744397), (25.566363100000004, -25.2203676), (25.2808114, -25.3123142), (28.4261398, -21.9980295), (27.8355177, -21.9683245), (25.580889300000003, -21.393204), (24.9580453, -23.6365031), (25.3220883, -24.256306100000003), (25.8506155, -24.8781975), (26.1385847, -24.3563596), (27.6657897, -23.0003379), (25.5642313, -24.6778541), (22.4038153, -25.9984402), (24.5352736, -25.6104999), (27.3760197, -20.4956895), (24.659291, -23.3814459), (27.0856832, -22.5665705), (27.7556473, -22.769962), (28.3385753, -21.785593), (28.8253293, -21.818867), (27.6265908, -20.9089548), (26.6791213, -22.380869), (27.0775043, -20.903558), (27.9457233, -21.452409), (27.3460497, -20.4908685), (22.1505772, -18.5255499), (27.5862343, -20.720187), (27.5475147, -20.5925245), (21.7260763, -21.060057), (27.9795423, -21.590858), (28.1103136, -21.738262), (27.395432300000003, -20.603234), (21.5678777, -18.4041095), (23.4249663, -20.019146), (24.7221565, -24.5987934), (25.4174335, -24.7916144), (25.8582544, -24.8900618), (25.9686877, -24.681206600000003), (21.2575537, -24.1357145), (25.3152243, -24.994663600000003), (25.5338567, -25.3132666), (21.6026802, -23.4121235), (26.785592600000005, -23.1227576), (25.3394829, -24.9982619), (26.896176300000004, -22.850745), (27.8644233, -21.9698771), (25.5815496, -21.4208125), (22.7762863, -23.6820871), (27.2433657, -21.0603965), (26.3742203, -20.1095999), (26.0986039, -24.3969571), (26.329368, -21.0904355), (21.7895173, -18.2953959), (27.3125097, -22.7181185), (24.4013323, -21.031999), (25.1613037, -17.8002464), (27.0654567, -20.499248), (27.4620453, -21.471365), (24.8654562, -24.044096), (21.511280300000003, -23.6610101), (26.238488300000004, -22.949211), (25.8947236, -24.6956076), (25.2494178, -24.7357848), (25.4350633, -25.4017341), (22.9128787, -18.6003155), (27.8235113, -21.979341), (21.8434003, -18.3892319), (23.3299989, -20.4253637), (28.3963644, -21.9689521), (21.781730300000003, -24.0147451), (25.8739675, -24.6512318), (27.3419393, -22.849266), (22.4192193, -26.0209072), (21.8953193, -18.4474535), (23.3877918, -19.99958), (25.5818883, -21.406083), (27.472611, -21.1955265), (26.032416100000003, -24.4227238), (22.3139545, -19.0160607), (21.6386383, -21.710385), (23.5874667, -19.8507795), (26.8032773, -23.1314331), (27.1315245, -22.5362651), (21.9578073, -18.3521869), (27.433238300000003, -21.392851), (22.1559583, -19.3756919), (26.6934723, -22.428694), (23.9998747, -19.1854135), (25.9098789, -24.6123472), (26.3847163, -22.600001), (28.4140243, -21.997759), (25.6752853, -23.359571), (25.5985647, -24.388469600000004), (24.4116943, -21.017539000000003), (21.872414300000003, -24.740905), (23.54523, -24.6372693), (24.6713179, -24.8307621), (25.8252881, -24.646823), (27.4540077, -21.186132), (27.5394577, -21.1650602), (22.8429077, -20.3795495), (26.1966044, -20.2102547), (27.2608217, -20.9445235), (23.3714313, -24.06029), (26.1579477, -24.3566496), (23.4522395, -20.0153), (27.134711300000003, -21.08133), (23.6223811, -24.3291942), (25.4915253, -25.5421891), (27.7795413, -21.483799), (26.7707657, -22.4388545), (23.4537617, -20.015205), (25.6932638, -25.186135), (24.4497217, -24.7816806), (25.2053843, -24.189700100000003), (22.3136833, -19.0163819), (25.3285597, -24.6495285), (25.7265207, -24.7634896), (25.6034667, -24.6924466), (27.9520053, -22.189163), (21.3000413, -18.97333), (25.9394211, -24.666548), (26.7534303, -21.679335), (21.9458047, -21.5189795), (26.8148963, -23.0950051), (25.3152243, -24.9945039), (22.4318743, -25.861596), (25.7466503, -25.024695), (25.9194035, -24.6552917), (25.4885712, -24.4095459), (26.1569008, -24.4014541), (25.0763291, -24.981935), (25.8758096, -24.7068247), (25.6864884, -25.2294473), (25.314906, -24.9948273), (25.5159831, -24.3754937), (26.1408158, -24.395007), (26.8352652, -23.0943731), (27.502235100000004, -21.1707082), (25.6886523, -25.2278581), (27.0684843, -20.902285), (23.4225029, -20.009715600000003), (25.6742978, -25.2043102), (27.6093183, -22.677748), (27.0536284, -22.562521), (25.8593112, -24.8742993), (21.6492333, -21.690554), (25.9781753, -24.6722902), (27.4469375, -21.4218256), (26.7452315, -22.413243), (27.4917285, -21.167272600000004), (25.8540594, -24.8653247), (26.8114042, -23.077343), (27.5033993, -21.1688223), (25.9082014, -24.6774137), (27.818256899999994, -21.9632398), (23.4012116, -19.9947495), (28.4175968, -21.991287600000003), (28.404926, -21.973558800000003), (26.7006183, -22.4316435), (25.5801953, -21.422796), (27.084158, -22.5578448), (25.4185009, -24.7629461), (25.1602489, -17.7879155), (22.786036300000003, -23.663496), (22.1570263, -19.3737129), (26.314265, -24.0231554), (25.7395238, -25.0289852), (27.5805453, -20.550011), (27.3334288, -21.0010358), (21.7829274, -23.9895918), (27.7568233, -22.797112), (27.7598343, -21.887355), (25.9374046, -24.621213), (25.3259483, -24.954864), (27.640605, -22.6077934), (27.854287300000003, -21.974967000000003), (26.6853703, -22.36917), (25.0922487, -25.729208), (26.8299445, -23.1077961), (24.4094437, -21.0284225), (27.5246143, -23.0295525), (25.5524117, -24.663003600000003), (26.4314803, -24.6022611), (25.588405, -25.4699211), (27.2622173, -20.80269), (26.4984492, -23.0581369), (27.439275800000004, -21.4203846), (27.4324303, -20.506094), (25.798124, -24.655755), (25.8857849, -24.6626128), (26.1723833, -24.3729141), (25.7020173, -24.6535521), (25.250078, -20.2093294), (27.585201300000005, -20.719751), (27.2323276, -20.652949600000003), (27.4547233, -20.6350597), (27.4899483, -21.147233), (24.7356266, -24.6049084), (27.2074063, -20.118701), (25.0232933, -24.117685100000003), (27.147825, -20.3007438), (25.3744283, -21.332399), (26.0942937, -24.3942144), (25.9086151, -24.6799269), (27.0252613, -20.492365), (25.5812888, -21.3977936), (26.2185833, -20.5627288), (21.8868003, -26.6547002), (25.5660653, -24.3948857), (23.563896300000003, -24.4002201), (27.8291889, -21.985391600000003), (27.1344663, -22.538896), (25.3463868, -25.010189), (28.4433675, -21.9837318), (25.5811601, -21.3935389), (25.8620887, -24.2442826), (22.6840043, -20.484906), (22.1589913, -18.7343979), (26.7393163, -22.38518), (21.8233963, -18.3878843), (26.209874300000003, -20.210294), (25.4253673, -24.8056977), (25.8556473, -24.8550887), (27.8678393, -21.980639), (26.0421249, -24.433319), (25.4656484, -24.3982619), (27.538712, -21.1685232), (25.5899263, -24.9800641), (27.5224685, -21.2072037), (27.4111848, -22.7242688), (25.407124300000003, -25.647101100000004), (26.7586784, -22.4358398), (25.8872814, -24.4859866), (27.4629021, -21.4599335), (25.691852800000003, -25.1990414), (25.889378, -24.6565025), (25.2559927, -25.307901), (25.8370465, -24.628510100000003), (22.7966255, -23.6656039), (26.8102723, -23.1371391), (25.932313, -24.6671812), (26.1412243, -24.418332100000004), (20.0772323, -22.2825576), (27.8165668, -22.3315428), (25.6683881, -24.7207153), (25.3700065, -24.9915766), (28.4031353, -22.422746), (20.8558447, -22.1705795), (23.4398323, -19.9954829), (25.4604173, -24.917288100000004), (27.2982213, -21.911796), (25.2480393, -20.209484), (22.2927033, -19.1262489), (28.7231303, -21.844221), (26.5658197, -22.940007), (27.5792675, -22.3880152), (27.819203100000003, -21.9796862), (20.2748103, -23.126406), (24.5520613, -25.756058), (25.7371353, -23.441907), (27.8825585, -21.3108846), (23.4152696, -19.9951167), (25.6992383, -24.987071), (27.522656, -21.2017802), (26.3942873, -22.642293), (25.946542, -24.6090005), (22.4174255, -18.8175672), (25.5566144, -25.3281701), (26.0439461, -24.5607228), (27.0625293, -20.525377), (27.534049300000003, -22.879372), (27.6336837, -20.8390725), (27.2079538, -20.8130872), (23.2818717, -25.2745136), (26.6791333, -22.417583), (25.5698822, -25.2086589), (27.1012893, -22.5611584), (27.3156713, -20.554196), (23.8839773, -20.330475), (25.9258454, -24.6393486), (25.4431813, -25.0844711), (27.570976300000005, -22.733051), (26.428601300000004, -21.864576), (27.2229489, -22.6570938), (28.589868300000003, -22.083273), (23.5013043, -24.713396), (26.669704100000004, -23.6854481), (27.487463, -23.1864543), (25.3014105, -24.9752168), (26.8154393, -22.226683), (27.7087707, -20.5700905), (27.4121973, -22.858572), (27.4476453, -21.452789000000003), (27.9594311, -22.1973157), (27.467350300000003, -21.296673), (25.9809675, -24.6613775), (27.1880173, -20.508525), (25.884189300000003, -24.610315300000003), (25.5128583, -24.6930778), (25.4160013, -25.4749811), (26.7219323, -22.365122), (26.8556392, -23.1122374), (25.694094, -25.185669), (23.4538617, -19.9584895), (25.837556, -24.7812109), (27.5511978, -21.1902387), (25.8135615, -24.8433537), (24.8930882, -21.2132182), (25.8962016, -24.6474776), (26.2130794, -20.5602145), (27.0009797, -22.7888715), (22.4571907, -20.1647465), (26.1620179, -24.2957295), (27.1287498, -23.4064704), (25.7959263, -24.555698100000004), (25.9488836, -24.6088882), (27.6711059, -22.9910397), (25.5443287, -24.3718436), (25.4234094, -24.7584768), (28.422757300000004, -21.998206), (26.469817300000003, -23.039002), (24.1356803, -20.218709), (27.84516810000001, -21.9967735), (26.705240000000003, -22.3986651), (26.167143300000003, -24.4151471), (25.8511519, -24.8809083), (21.8502393, -23.916444100000003), (24.7053233, -23.699901), (25.4673753, -24.422119), (24.6293053, -24.894411), (26.5674293, -22.427217), (25.528981100000003, -24.3596907), (25.7847507, -24.6855986), (26.8387038, -23.1513399), (25.888407, -24.6180163), (25.8648865, -24.6545943), (25.2815698, -23.8366366), (27.4956027, -21.1340255), (26.1483663, -24.3509201), (24.7211127, -24.6111188), (24.5112143, -18.1474459), (28.046086, -21.9225397), (25.862674, -24.6650398), (25.8847976, -24.6267844), (27.729690500000004, -21.8712587), (28.4067285, -21.9733716), (25.3086387, -24.9525587), (26.0333223, -21.22282), (27.4492162, -20.6420749), (25.4610865, -24.409915), (21.6492523, -21.682273), (27.0419292, -20.4911266), (26.5208516, -23.0610085), (25.409626100000004, -25.4647264), (27.735819300000003, -21.9033), (23.4679233, -19.9792979), (26.2028773, -20.206971), (27.3365304, -21.0952638), (27.8782027, -21.964533), (27.4725745, -21.2051699), (25.8521763, -24.620157), (21.8370693, -18.3568449), (21.8626947, -18.2940615), (25.8777421, -24.6760582), (24.3070023, -24.015704100000004), (22.4190533, -26.0208542), (24.7327764, -24.6034837), (26.7569855, -22.4340952), (25.4612583, -24.3976842), (25.916454, -24.6057989), (25.4826784, -24.383026), (22.0753956, -18.4540537), (25.8963795, -24.6511562), (25.3605813, -21.338597), (25.788913600000004, -24.6223789), (25.3647983, -24.4139839), (27.2804469, -20.8998741), (25.3728223, -21.337855), (25.9491463, -24.6116261), (25.9427633, -24.6293522), (27.5155432, -21.1581269), (25.6635287, -25.2351895), (26.8577027, -23.1040075), (23.4511975, -19.9900088), (26.2256173, -20.563896), (25.3511517, -24.988156600000003), (25.9028801, -24.6112123), (25.5819448, -24.6580269), (25.9006302, -24.6350449), (27.5100791, -21.165394600000003), (27.0377717, -22.5305707), (25.4560811, -24.4067081), (25.3160023, -24.9939021), (27.8340716, -21.979772), (23.4519807, -20.0359992), (25.8495073, -24.6324732), (25.7018777, -25.2290646), (25.8793611, -24.6717929), (26.5044988, -23.0655322), (25.3613044, -21.3322023), (23.4370674, -20.0328276), (27.5420407, -21.1908859), (27.8474406, -21.971764800000003), (27.832212800000004, -21.9824458), (25.8885235, -24.6330737), (23.4491905, -19.990095), (25.9739854, -24.569831), (21.6468878, -21.687117800000003), (25.8704213, -24.6639851), (25.8413605, -24.869046), (25.8605236, -24.6584403), (25.4874627, -24.4128566), (27.4606673, -21.451327), (25.9776943, -24.6702031), (26.7377658, -22.4151414), (25.8785458, -24.634205), (25.9136298, -24.655230300000003), (27.5099128, -21.167805), (25.9615381, -24.6788672), (25.8837474, -24.677805600000003)]

    # Copernicus hub likes polygons in lng/lat format
    return Polygon([(y, x) if swap_coordinates else (x, y) for x, y in tln_points])


username = "kelsey.doerksen@keble.ox.ac.uk"
password = "GEPZRWb8e5$/a$7"

hub = SentinelAPI(username, password, "https://scihub.copernicus.eu/dhus")

data_products = hub.query(
    get_tallinn_polygon(swap_coordinates=True),  # which area interests you
    date=("20200101", "20200420"),
    cloudcoverpercentage=(0, 10),  # we don't want clouds
    platformname="Sentinel-2",
    processinglevel="Level-2A"  # more processed, ready to use data
)

data_products = hub.to_geodataframe(data_products)
# we want to avoid downloading overlapping images, so selecting by this keyword
data_products = data_products[data_products["title"].str.contains("T35VLF")]

hub.download("16082e6b-b32c-4cdc-8e0d-3d64f2432b88", "/Users/kelseydoerksen/Desktop/sentinel/src_data")


'''
src_root_data_dir = "/Users/kelseydoerksen/Desktop/sentinel/src_data"
tiff_root_data_dir = "/home/mike/sentinel/tiff_data"

bands = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
resolutions = ["R10m", "R10m", "R10m", "R10m", "R10m", "R10m", "R10m", "R10m", "R10m", "R10m"]
bands_and_resolutions = list(zip(bands, resolutions))

target_dim = (128, 128)


def extrapolate(arr, target_dim):
    x = np.array(range(arr.shape[0]))
    y = np.array(range(arr.shape[1]))
    z = arr
    xx = np.linspace(x.min(), x.max(), target_dim[0])
    yy = np.linspace(y.min(), y.max(), target_dim[1])

    new_kernel = RectBivariateSpline(x, y, z, kx=2, ky=2)
    result = new_kernel(xx, yy)

    return result
'''