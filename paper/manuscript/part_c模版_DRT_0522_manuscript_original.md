# Smart predict-then-optimize for dynamic meeting-point of many-to-one demand-responsive transit service

Jin Ana, Tian-Liang Liua,1, Guanghui Fua

aMoE Key Laboratory of Complex System Analysis and Management Decision, School of Economics and Management, Beihang University, Beijing 100191, China

# Abstract

Demand-responsive transit (DRT) can serve as a valuable addition to traditional fixed-route public transportation, yet a door-to-door service model typically comes with steep operating expenses, lengthy detours, and inconsistent service reliability. We examine a many-to-one DRT framework where travelers arrive sequentially throughout the booking window and have the option between home pickup and nearby meeting points. We introduce Dynamic meeting-point Recommendation and Pricing Optimization (DRPO), a tailored adaptation designed specifically for DRT systems, which employs an SPO-based decision-focused training approach. When processing each request, the operator presents passengers with both home pickup service and a selection of meeting locations, along with the associated fare, walking distance, and predicted in-vehicle travel time. The problem is formulated as a Markov decision process. The prediction component is trained using a mix of Huber and SPO+ loss functions to ensure that cost estimates are in sync with subsequent pricing decisions. We test the performance of DRPO in a synthetic data set and a semi-real Beijing scene, and compared it with the other five strategies. In the synthetic data, the average total cost of DRPO is $1 2 . 1 4 \%$ lower than that of Static-pricing. In the Beijing case, DRPO increased the net profit in all five test cases, which was $1 . 3 5 \%$ lower than the average total cost of DSPO, and also reduced the passenger abandonment rate from $4 . 0 4 \%$ to $3 . 2 4 \%$ . Generally speaking, these results show that DRPO can improve the operation performance of many-to-one DRT system without affecting passenger participation.

Keywords: demand-responsive transit; DRT; dynamic pricing; smart-predict-then-optimize (SPO); meeting point recommendation

# 1. Introduction

Demand-responsive transit (DRT) fills the gap between traditional scheduled public transportation and private car travel. DRT adjusts the service according to the actual needs of passengers, so as to improve the travel convenience in places where the travel mode is scattered, changing greatly and not dense enough to support the traditional fixed-line bus. In practical, many-to-one DRT is often used, such as commuters from suburban residential areas to work places in the city center, or providing school bus service to pick up students from different places to a school.

On-demand bus system usually has two service modes. One is home-pickup, and the bus will pick them up directly from each passenger’ home location. The other is meeting-point, where passengers need to walk to a recommended meeting point, and the bus will only stop when someone makes an appointment. Home-pickup is more comfortable for passengers, but it often leads to longer routes, more vehicles running, and less reliable schedules. Although meeting point can reduce detours and simplify the process, it may make the service less convenient and passengers will not want to use it. Existing studies have found that as long as the walking distance is not too far, this meeting point mode can actually make the service more reliable and efficient (Zheng et al., 2019; Zhang et al., 2023). However, the current literature offers limited direction regarding how service providers should coordinate operations when passengers have the flexibility to select between home pickups and meeting points.

The importance of this gap lies in that boarding decisions will affect the efficiency of operator and the experience of passenger. Studies from on-demand transportation and similar service industries show that whether passengers accept the service depends on the balance between the service details provided by price, walking distance and booking (Bai et al., 2019; Yan et al., 2025). But in practice, this information must be told to passengers at the time of booking, although the final routing plan can not be determined until all the requests are summarized. Therefore, operators face a continuous challenge: current pricing and displayed information will affect future routing decisions under uncertain demand. In order to solve this problem, this paper proposes a framework called Dynamic Meeting-Point Recommendation and Pricing Optimization (DRPO), which is specially used to handle many-to-one DRT services. DRPO is not a brand-new pricing system. It actually transforms the existing DSPO method and makes it available in the DRT scenario. This scene includes picking up people at home, meeting at the assembly point, and passengers may cancel their orders temporarily. At the same time, it also added a learning goal based on SPO, which pays more attention to the final decision. When a new passenger places an order, DRPO will first list a list of feasible options, including home pick-up and some nearby assembly points, and then set

a unique price for each option, so as to manage the demand and make the operation more efficient. This framework consists of three key parts: a method to generate the candidate list of assembly point according to the distance, a passenger selection model (this model will consider the passenger's reaction to the price, walking distance and driving time in the car), and a decision-oriented learning method (this method optimizes not the accuracy of prediction, but how helpful the prediction module is to the later pricing results).

This paper introduces three important breakthroughs: (i) It establishes a many-to-one Dynamic Ridepooling (DRP) pricing model, in which operators can coordinate door-to-door pick-up and heuristic meeting point selection, and can also handle the situation that passengers arrive one after another; (ii) It puts forward a set of practical DRPO framework, including distance-based candidate route generation rules, calibrated service time and utility parameter estimation, and upgrading DSPO; with SPO optimized learning scheme; (iii) It has carried out a lot of experimental verification, including six simulation benchmark strategies, five seed analysis in Beijing, matching seed comparison with DSPO, fleet size verification research, and sensitivity test of behavior factors.

This paper is structured as follows. Section 2 reviews related literature. Section 3 formulates the manyto-one DRT pricing problem. Section 4 details DRPO framework. Section 5 presents empirical results, and Section 6 concludes with policy implications and directions for future research.

# 2. Related work

First, in Section 2.1, we introduce recent research on DRT systems, including DRT, micro-transit and Customized bus (CB). Second, in Section 2.2, we discuss related work from other fields. Third, in Section 2.3, we summarize the research gaps and contributions of our study.

# 2.1. Demand-responsive transit systems

In recent years, the research of DRT system has made great progress, mainly focusing on system design, operation optimization and service equity. In system design, Baier et al. (2024) establish a four-step evaluation framework to systematically measure the performance of DRT; Sørensen et al. (2021) study the completely flexible door-to-door DRT service in rural Germany, and found that this service is usually concentrated on the main traffic routes, but it is inefficient in areas with low demand. Currie and Fournier (2020) comprehensively analyze the implementation of DRT in the past 40 years, and found that the failure

rate was high ( $50 \%$ in seven years), mainly due to the high cost, especially the complicated many-to-many service model.

In the aspect of operation optimization, data-driven and algorithm methods are widely used to improve the efficiency and adaptability of DRT. Lyu et al. (2019) and Xia et al. (2022) develop route planning frameworks for customized bus systems. These frameworks combined demand clustering with dynamic planning and differential evolution algorithm, which improved the operational efficiency by as much as $47 \%$ . Wu et al. (2024) introduce a multi-agent deep reinforcement learning framework to real-time control the route of customized bus networks, which can effectively reduce operating costs without sacrificing service quality. Kostic et al. (2021) propose a two-step deep survival modeling method for electric carsharing systems, which combined advanced neural network and spatial correlation modeling based on copula to improve idle time prediction and vehicle scheduling strategy. Lee et al. (2021) develop a twostage stochastic programming model for regional flexible public transport systems, which optimizes vehicle routing and ensures service reliability under demand uncertainty. Liu et al. (2023) utilize machine learning and spatial analysis to study the passenger volume of customized public transport in Shanghai, demonstrating that customized public plays a complementary role to the metro in suburban areas.

Equity and service quality have become increasingly prominent in recent DRT research. Liezenga et al. (2024) shows that the micro-transit systems in the suburbs especially benefits vulnerable populations, and $5 5 . 5 \%$ of the passengers are low-income people, thus improving the convenience of travel in underserved areas. Miller et al. (2016) contribute a tool called Public Transit Sustainability Mobility Analysis Tool (PTSMAT), which combines environmental, economic, social and system efficiency indicators to comprehensively evaluate the sustainability of public transport systems. Shang et al. (2022), use smart card data from Beijing and find that, although customized bus systems can alleviate congestion and improve commuting comfort, their higher fares may limit accessibility for low-income passengers. Other studies, such as Knierim and Schlüter (2021), reveal that in rural Germany, DRT adoption is negatively correlated with car ownership and positively correlated with physical disability and old age. This highlights the important role of DRT in mitigating travel inequality for vulnerable groups.

Incentive strategies and pricing models are also gaining attention for improving DRT performance. Wang et al. (2021) develop a bi-level optimization framework that combines passenger behavioral modeling with vehicle route planning. They found that targeted monetary incentives could reduce detours by $23 \%$ and increase operators' profits by $1 8 \%$ . Li et al. (2021) employ a game-theoretic approach to optimize pricing for customized bus and ride-sharing services. They show that strategic price

differentiation based on value of time and vehicle ownership can enhance platform profitability. Liu and Ceder (2015) point out that policy coordination and system integration remain key challenges for the largescale implementation of customized bus services in China. Although a few studies have examined how pricing can improve operator efficiency in DRT systems, most of them do not fully consider the sequential arrival of passengers and the spatiotemporal distribution of demand.

Recent research on DRT and on-demand ride-sharing has found that passengers can be flexible in time and place, which is very helpful for improving the efficiency of the entire system. Wu et al. (2025) and Zhang et al. (2025) propose new operating strategies, such as dynamic stop assignment and joint supply– demand management, to reduce the high cost and detours that are inherent in traditional door-to-door DRT systems, especially when they are used as feeder services to rail transit. Fielbaum et al. (2021) and Li et al. (2025) further show that encouraging passengers to walk to optimized meeting points or flexibly delaying matches can greatly improve operational efficiency and user satisfaction. Generally, these works highlight that integrating intelligent assignment, dynamic pricing, and flexible user behavior leads to more sustainable, adaptive, and efficient DRT systems.

# 2.2. Meeting points, walking coordination, and pricing

The literature demonstrates advanced approaches to dynamic pricing across industries. Galiullina et al. (2024) and Akkerman et al. (2024) propose stochastic optimization frameworks integrating machine learning with two-stage stochastic programming for last-mile dynamic pricing. In retail supply chains, Kayikci et al. (2022) propose an IoT-driven dynamic pricing model that adjusts food prices in real-time based on freshness indicators, reducing perishable waste by $4 7 . 6 \%$ compared to fixed pricing strategies, while Dong et al. (2009) and Kouvelis et al. (2008) derive optimal pricing policies for substitutable products under inventory constraints. For service industries, Yang et al. (2013) found that the pricing strategy combined with the route planning can increase the profit of fresh e-commerce by $3 . 8 \%$ . The literature shows a key finding: dynamic pricing can increase revenue in many industries such as transportation and e-commerce through price adjustment based on demand.

Meeting-point-based operations have received growing attention because they offer a way to improve routing efficiency without abandoning service flexibility altogether. Early work by Stiglic et al. (2015) showed that introducing meeting points into ride-sharing systems can substantially reduce vehicle mileage while keeping walking distances manageable. Zheng et al. (2019) show that introducing meeting points into flex-route transit can generate sizable operational benefits, and Fielbaum et al. (2021) demonstrate

that optimized walking locations can improve on-demand ridesharing performance. More recent work reaches similar conclusions in richer on-demand settings. Wu et al. (2025) study dynamic DRT scheduling with time-dependent travel times, Zhang et al. (2025) examine fixed-route and demand-responsive transit design with dynamic stops, Li et al. (2025) analyze delayed matching and walking in on-demand ridesharing, Pellegrini and Fielbaum (2025) study users' willingness to accept flexible walking, and Yan et al. (2025) investigate the adoption trade-off between dynamic and static walking. Together, these studies show that flexible stop assignment and walking-based coordination can improve efficiency in on-demand systems. Even so, most studies treat meeting-point assignment as a planning or control problem and pay limited attention to how real-time prices can be used to influence passenger acceptance of meeting points relative to home pickup.

Dynamic pricing has long been studied as a mechanism for balancing demand and operational costs under uncertainty. Gallego and van Ryzin (1994) develop a classical inventory-based model showing how prices can be adapted over finite horizons under stochastic demand. In logistics and delivery settings, Galiullina et al. (2024) and Akkerman et al. (2025) show that dynamic selection and pricing can steer demand toward more efficient service options. Yang et al. (2016) demonstrate that pricing and routing decisions can be profitably coordinated in e-fulfillment, while Dong et al. (2009) characterize optimal pricing under substitution effects. Beyond transportation, Kayikci et al. (2022) propose a data-driven dynamic pricing strategy for reducing perishable food waste, Chen et al. (2017) study dynamic pricevector formation for demand response in photovoltaic-assisted electric-vehicle charging stations, and Zhao and Lee (2022) develop a deep-reinforcement-learning approach for dynamic pricing at electricvehicle charging stations. In DRT and adjacent mobility services, pricing has also been identified as a promising lever. Wang et al. (2021) show that passenger incentives can reduce detours and improve operator profit, and Li et al. (2021) study strategic pricing in customized bus and ride-sharing settings. Taken together, the literature shows that dynamic pricing can improve revenue or operating performance in several industries by adjusting prices to demand, costs, and service conditions. Yet most of this work remains static or offline in nature, and it rarely models the sequential arrival of passengers together with the need to quote service attributes, such as in-vehicle travel time, before the downstream routing problem is fully realized.

# 2.3. Passenger behavior and decision-focused learning

Equity and user experience remain important concerns in the DRT literature. Liezenga et al. (2024) evaluate access equity in suburban microtransit, Shang et al. (2022) examine the integration of conventional and customized bus services in Beijing, and Knierim and Schlueter (2021) study attitudes toward demand-responsive transport among potentially less mobile people in rural Germany. These access-equity evaluations and empirical studies of customized bus services show that service design affects who benefits from flexible transit and how willing different user groups are to adopt it. These findings highlight the importance of explicitly modeling behavioral responses when designing demandsteering mechanisms in DRT.

The methodological challenge of the present problem is not only to price options online, but also to learn cost predictions that are useful for the downstream pricing decision. This connects the problem to decision-focused learning. Elmachtoub and Grigas (2022) show that prediction models should ideally be trained with respect to the induced downstream optimization loss rather than a purely pointwise prediction loss. Their Smart Predict-Then-Optimize framework provides exactly this perspective. Donti et al. (2017) extend decision-focused learning to task-based end-to-end model learning, and Wilder et al. (2019) develop decision-focused learning methods for combinatorial optimization settings. In platform-based mobility, Yang et al. (2025) apply an SPO-style decision-focused learning framework to ride-hailing subsidy allocation. Mandi et al. (2024) provide a broader survey of decision-focused learning methods and benchmarks. In the present setting, the relevant downstream problem is a pricing problem defined over predicted marginal insertion costs. The paper therefore extends the logic of decision-focused learning to a many-to-one DRT context with meeting-point candidate generation and endogenous passenger choice.

# 2.4. Research gaps and contributions

Our work is motivated by three gaps in the literature. First, although meeting points and demand steering have both been studied in DRT, the literature has paid limited attention to settings in which passengers may choose between home pickup and meeting points in response to real-time prices. Second, the majority of DRT optimization studies maintain a static, offline, or routing-centric approach, failing to adequately represent the sequential pattern of passenger arrivals. Third, even when predictive models are implemented, they are commonly optimized for pointwise accuracy rather than for the actual quality of subsequent operational decisions.

We fill in the gaps with a DRPO framework. We study a many-to-one DRT model where the operator presents passengers with a set of boarding choices, each tailored to their preferences for price, walking

distance, and in-vehicle travel time. The predictive model is fine-tuned through SPO objective that aligns with the subsequent pricing decision. The proposed framework integrates behavioral analysis, dynamic pricing, and decision-focused learning into a unified, sequential decision-making process.

# 3. Model

This paper studies a many-to-one demand-responsive transit service in which passengers travel from dispersed origins to a common destination and submit requests sequentially during the booking horizon. For each request, the operator prices a menu of service options, including home pickup and recommended meeting points, before the full demand realization and final routing plan are known.

# 3.1. Problem description

In this particular scenario, passenger demand is unpredictable and revealed online. Whenever a new passenger joins the reservation system, the operator gets their home location and quickly suggests a couple of potential boarding locations. This lineup includes the passenger's home location and a few suggested meeting points. Each option is paired with a fare, a walking distance, and a predicted in-vehicle travel time. Subsequently, the passenger can choose one of these options or opt out of the service altogether.

The operator cleverly employs pricing tactics to shape passenger choices, offering discounts to those who walk to meeting points while imposing extra fees on passengers who demand home pickup. This strategy aims to strike a balance between maintaining passenger acceptance and minimizing routing burden generated associated with multiple individual pickups. Fig. 1 contrasts the two boarding modes considered in the paper.

![](images/b6337f2de9a203cac6b7bbf5386966e34dc55ace832122b83886e5a1070ac941.jpg)  
(a) Home pickup

![](images/aa3b0fefd2e0a404cb2d186730c629f085cac1353ab179ddee6986dd9a459840.jpg)  
(b) Walking to recommended meeting points

Fig. 1. Home pickup vs. walking to recommended meeting points in a many-to-one DRT system

Since the final routes for vehicles are locked in only after all bookings are finalized, the system has to gauge the practical implications of each choice made, taking into account the unpredictability of demand. Fig. 2 shows the passenger interface in our DRT system.

![](images/15dbec3398a29790659de25940a9e7e0b71bc38823a4f5f9a2dd92dd17a53399.jpg)  
Fig. 2. Passenger interface

# 3.2. Problem formulation

# 3.2.1. Decision timeline

We simulate the booking process as a series of distinct decision periods within a defined booking interval, $[ 0 , 1 , \dots , T ]$ , where $T$ represents the deadline for bookings. Once the time $T$ has elapsed, no more requests are taken; the operator then addresses the multi-vehicle routing using the actual bookings made during that period. Passenger requests arrive sequentially at epochs $t = 1 , \dots , T$ , with origins drawn from the service-area distribution. At each epoch $t$ , a passenger arrives with probability $\lambda _ { t } \in [ 0 , 1 ]$ and no passenger arrives with probability $1 - \lambda _ { t }$ .

At each epoch, the online interaction unfolds as follows, shown in Fig. 3.

1. A passenger request arrives at epoch $t$

2. The system recommends several boarding options (including home pickup and meeting points), each with an associated price, walking distance and in-vehicle travel time.   
3. The passenger selects an option $k \in O _ { t }$ or rejects the service.   
4. After the cutoff time $T$ , the operator performs multi-vehicle routing optimization.

Passenger accesses system at time t.   
System shows boarding options.

Homet

M-P1 1 fare:12.5￥ i-v-t:32 min walk:300m

M-P², t fare:9￥ i-v-t: 26 min walk:500 m

P,selects $\mathrm { M - P _ { \mathrm { ~ t ~ } } ^ { 2 } }$

M-P2t t √ selected

![](images/86b8033a97e5f7b44291626d4a7304d55ca942b08a601d16249ad9a82ad39f9d.jpg)  
Fig. 3. Online DRT system workflow

For passenger $t$ , the booking system offers home pickup $h _ { t }$ and a set of recommended meeting points $M _ { t }$ , which together form the complete range of choices $O _ { t } = \{ h _ { t } \} \cup M _ { t }$ . For example, if ${ { M } _ { t } } \mathrm { ~ = ~ }$ $\{ m p _ { 1 } , m p _ { 2 } , m p _ { 3 } \}$ , then $O _ { t } = \{ m p _ { 1 } , m p _ { 2 } , m p _ { 3 } , h _ { t } \}$ . In this work, $M _ { t }$ is determined heuristically by choosing the $n$ nearest meeting points to the passengers' homes. The home pickup option is always part of the deal, ensuring that passengers can still opt for home pickup.

# 3.2.2. Net profit function

The operator aims to boost its earnings by redirecting some passengers from home pickup to designated meeting points, thus cutting down on transportation expenses without excessive compromising the quality of service. The operating cost contains two components: i) labor cost associated with vehicle operating time, denoted by $c _ { \omega }$ ; ii) fuel cost associated with travel distance, denoted by $c _ { f }$ ; Here, the finalized route plan $R ^ { T }$ is defined to include the final leg from the last served location to the common destination.

Therefore, this final leg is already incorporated into the route-arc set ${ \mathcal { E } } ( R ^ { T } )$ , and its associated costs are accounted for in Eq. (2).

Passenger requests arrive sequentially, and the total number of arrivals follows an unknown demand process $D$ . To capture behavioral heterogeneity, we divide passengers into segments $g \in { \mathcal { G } }$ , where $\mu _ { g }$ denotes the probability that an arriving passenger belongs to segment $g$ .

Once the booking deadline hits, the finalized bookings and route plan become clear. Let $B _ { T }$ denote the accepted booking set at the close of the booking period, and $R ^ { T }$ denote the associated route plan. The operator’s objective is to maximize total net profit:

$$
\max  \Pi \left(B _ {T}\right), \tag {1}
$$

The downstream cost component of $\Pi ( B _ { T } )$ is denoted $C _ { T } ( B _ { T } )$ , so that $\Pi ( B _ { T } ) = \mathrm { R e v e n u e } ( B _ { T } ) -$ $( B _ { T } ) -$ $C _ { T } ( B _ { T } )$ , where R $\begin{array} { r } { ( B _ { T } ) = \sum _ { ( t , k ) \in B _ { T } } ( f _ { e } + a _ { t k } ) } \end{array}$ and

$$
C _ {T} (B _ {T}) = c _ {\omega} \left(\sum_ {(i, j) \in \mathcal {E} (R ^ {T})} \omega_ {i, j} + \sum_ {i \in \mathcal {N} (R ^ {T})} l _ {i}\right) + c _ {f} \sum_ {(i, j) \in \mathcal {E} (R ^ {T})} d _ {i, j}. \tag {2}
$$

In R $( B _ { T } ) , f _ { e }$ is the base fare and $a _ { t k }$ is the option-specific price adjustment. $a _ { t k } < 0$ represents a discount, whereas $a _ { t k } > 0$ represents a surcharge.

In Eq. (2), the first term represents the time-related operating cost, computed as the unit time cost $c _ { \omega }$ multiplied by the total vehicle operating time. Specifically, $\begin{array} { r } { \sum _ { ( i , j ) \in \mathcal { E } ( R ^ { T } ) } \omega _ { i , j } } \end{array}$ is the total travel time over all route arcs, and $\textstyle \sum _ { i \in { \mathcal { N } } ( R ^ { T } ) } l _ { i }$ is the total dwell time at visited nodes. The second term encompasses the operating cost tied to distance, calculated by multiplying the unit cost per distance, denoted as $c _ { f }$ , with the total distance traversed across all route segments. Since $R ^ { T }$ includes the final leg from the last served location to the common destination, the relevant time and distance cost are already included in the second and third components.

# 3.3. Dynamic programming modeling and the Bellman equation

The online booking problem is inherently dynamic. Before the deadline $T$ , the operator cannot predict how many extra passengers might show up or how those future requests will affect the current routing. The choice of boarding location for a given passenger directly impacts the system's future configuration, potentially reshaping how subsequent routes are arranged. This intricate relationship between current decisions and future outcomes lends itself naturally to a Markov decision process (MDP) framework.

Let $V _ { t } ( s _ { t } )$ represent the highest anticipated aggregate profit from epoch $t$ onwards until the end of the booking period, based on the state $s _ { t }$ . Since routing costs are only incurred after the booking horizon is finalized, the terminal value is determined solely by the ultimate routing cost:

$$
V _ {T + 1} (s _ {T + 1}) = - C _ {T} (B _ {T}). \tag {3}
$$

At epoch $t$ , the system state is written as

$$
s _ {t} = B _ {t - 1}, \tag {4}
$$

where $B _ { t - 1 }$ represents accepted booking set before time period ??. When a new passenger makes a request, the system captures $g _ { t }$ , the home location.

Once the passenger arrives, the recommendation system generates the option set $O _ { t }$ , and the operator picks price-adjustment vector $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ . After that, the passenger's decision-making process follows the multinomial logit model, which is detailed in Section 3.4. If the passenger selects an internal option $k \in O _ { t }$ , the state is updated to $s _ { t + 1 } = s _ { t } \cup \{ ( t , k ) \}$ ; otherwise, it stays the same $s _ { t }$ .

Because terminal value accounts for downstream routing costs, the immediate reward at time $t$ is solely derived from the fare of the current passenger. Let $f _ { e }$ denote the base fare. Then the expected one-period reward for a passenger is

$$
r _ {t} \left(\boldsymbol {a} _ {t}\right) = \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right). \tag {5}
$$

If passenger choose the outside option, or if no passenger arrives, the immediate reward is zero. The Bellman optimality equation becomes

$$
V _ {t} (s _ {t}) = \max  _ {\boldsymbol {a} _ {t}} \left\{\lambda_ {t} \left[ \begin{array}{c} \sum_ {k \in O _ {t}} P _ {t k} (\boldsymbol {a} _ {t}) \cdot \left[ \left(f _ {e} + a _ {t k}\right) + V _ {t + 1} \left(s _ {t + 1}\right) \right] \\ + P _ {t 0} (\boldsymbol {a} _ {t}) \cdot V _ {t + 1} \left(s _ {t}\right) \end{array} \right] + (1 - \lambda_ {t}) V _ {t + 1} \left(s _ {t}\right) \right\}. \tag {6}
$$

Cracking Eq. (6) head-on is computationally demanding due to the state space ballooning as booking patterns evolve and future arrivals remain uncertain. Even when employing a fixed heuristic to generate the option set, the routing implications of any given decision hinge on future demand fluctuations. Consequently, we turn to approximate dynamic programming, reframing each boarding option's downstream value by assessing its marginal impact on overall routing costs. This approach yields a manageable per-passenger expected-profit formulation, detailed in Section 4.

# 3.4. Passenger choice model

Passengers' behavior is analyzed using a multinomial logit model (MNL). Once a passenger arrives at epoch $t$ , the operator presents them with a set of options denoted as $O _ { t }$ , along with the associated price vector $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } } .$ . Next, the passenger selects either one of the available options within the set or quit.

Passenger preferences are characterized by positive disutility weights for walking distance, price, and in-vehicle travel time, with corresponding parameters ??walk, ??price, and $\beta ^ { \mathrm { i v t } }$ . The price term enters utility as $- \beta ^ { \mathrm { p r i c e } } a _ { t k }$ with $\beta ^ { \mathrm { p r i c e } } > 0$ , so a positive price adjustment lowers utility and a negative adjustment (a discount) raises utility. The outside option is characterized by a common baseline utility constant $V _ { 0 }$ .

For a passenger, the utility of internal option $k \in O _ { t }$ is

$$
U _ {t k} = - \beta^ {\mathrm {i v t}} \tau_ {t k} - \beta^ {\mathrm {w a l k}} d _ {t k} - \beta^ {\mathrm {p r i c e}} a _ {t k} + \varepsilon_ {t k}, \tag {7}
$$

where $\tau _ { t k }$ is the predicted in-vehicle travel time, $d _ { t k }$ is the walking distance, and $a _ { t k }$ is the option-specific price adjustment. Given that $\beta ^ { \mathrm { i v t } } > 0 , \beta ^ { \mathrm { w a l k } } > 0$ $\beta ^ { \mathrm { i v t } } > 0$ , and $\beta ^ { \mathrm { p r i c e } } > 0$ , shorter travel time, shorter walking distance, and a lower price all contribute to higher utility.

The disturbance terms $\varepsilon _ { t k }$ and $\varepsilon _ { t 0 }$ are posited to be independently and identically distributed following the Gumbel distribution, which gives rise to the conventional multinomial logit (MNL) framework. The outside option for a passenger is captured by

$$
U _ {t 0} = V _ {0} + \varepsilon_ {t 0}. \tag {8}
$$

The probability that a passenger chooses internal option $k$ is

$$
P _ {t k} \left(\boldsymbol {a} _ {t}\right) = \frac {\exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t k} - \beta^ {\mathrm {w a l k}} d _ {t k} - \beta^ {\mathrm {p r i c e}} a _ {t k}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O _ {t}} \exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t j} - \beta^ {\mathrm {w a l k}} d _ {t j} - \beta^ {\mathrm {p r i c e}} a _ {t j}\right)}, \tag {9}
$$

and the probability of choosing the outside option is

$$
P _ {t 0} \left(\boldsymbol {a} _ {t}\right) = \frac {\exp \left(V _ {0}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O _ {t}} \exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t j} - \beta^ {\mathrm {w a l k}} d _ {t j} - \beta^ {\mathrm {p r i c e}} a _ {t j}\right)}. \tag {10}
$$

These probabilities satisfy

$$
P _ {t 0 g} (\boldsymbol {a} _ {t}) + \sum_ {k \in O _ {t}} P _ {t k g} (\boldsymbol {a} _ {t}) = 1. \tag {11}
$$

The MNL model bridges the gap between pricing and routing outcomes. By offering larger discounts for meeting-point options and applying surcharges for home pickup, the operator can leverage prices to steer boarding decisions, thereby influencing the downstream routing cost.

# 4. Methodology

The DRPO framework combines a heuristic recommendation module, a predictive module for marginal insertion costs and pickup positions in the service sequence, and a pricing module trained with a decisionfocused objective. The goal is to approximate the long-run routing impact of current decisions while preserving a tractable online implementation.

# 4.1. Framework overview

The main hurdle in dynamic routing pricing is that the true marginal insertion cost of assigning a passenger to a particular boarding location is not just a simple calculation; it's influenced by future passengers and the ultimate route, both of which are unpredictable at the time a price needs to be set. DRPO helps mitigate this issue by breaking it down into offline and online components. Online, DRPO follows a recommend-predict-price pipeline. First, a heuristic recommendation module constructs the option set $O _ { t }$ by retrieving the precomputed adjacency list of the 5 closest feasible meeting points for the current home location and adding the home-pickup option. Second, for each candidate option $k \in O _ { t }$ , the prediction module maps the option-level encoding $\phi ( s _ { t } , k )$ to (i) an estimated marginal downstream cost proxy and (ii) a predicted in-vehicle travel-time proxy $\widehat { q } _ { t k }$ , which is converted into the displayed in-vehicle travel time $\tau _ { t k }$ . Third, conditional on the offered set and the predicted marginal costs, the pricing module determines the option-specific price vector $\mathbf { a } _ { t }$ to maximize expected profit. Offline, the same framework generates counterfactual routing labels and trains the prediction model with a blended pointwise and decision-focused objective.

Fig. 4. Overview of the DRPO framework. The online layer recommends feasible boarding options, predicts option-level routing impact, and solves the pricing problem for the current passenger. The offline layer generates counterfactual routing labels and updates the predictive model with pointwise and decision-focused losses.

To obtain a tractable online decision problem, the dynamic program is reformulated as a per-passenger expected-profit problem in which the routing impact of each candidate option is represented by its marginal cost. Following Yang et al. (2016), the derivation from the Bellman equation is presented in Appendix B. Algorithm 1 summarizes the implementation workflow used in the experiments.

# Algorithm 1. Offline-online training and deployment procedure of DRPO

1. Precompute, for each home location, the adjacency set of the 10 nearest meeting points and initialize the choice-model and cost parameters.   
2. Simulate booking episodes under the calibrated demand model and, for each decision epoch ??, enumerate the feasible boarding options $O _ { t } = \{ h _ { t } \} \cup M _ { t }$ .

3. For every candidate option $k \in O _ { t }$ , construct the option-level encoding $\phi ( s _ { t } , k )$ . To generate the counterfactual routing label, first form the baseline terminal booking set $B _ { T } ^ { - t }$ by removing passenger ??s realized accepted booking if present, and then form $B _ { T } ^ { t , k } = B _ { T } ^ { - t } \cup \{ ( t , k ) \}$ . The label is computed as $c _ { t k } = C _ { T } ( B _ { T } ^ { t , k } ) - C _ { T } ( B _ { T } ^ { - t } )$ using the same terminal routing solver, namely a greedy nearest-insertion heuristic with 2-opt post-processing. For each passenger, this requires one shared baseline routing evaluation and one counterfactual routing evaluation per candidate option, taking approximately 0.3,s per solve on the experimental hardware.   
4. Train the marginal-cost predictor with a Huber loss during the initial phase; once the representation stabilizes, ramp up the $\mathrm { S P O + }$ term and optimize the blended objective in Eq. (eq:blend-loss).   
5. At deployment time, retrieve the feasible meeting-point set for the current passenger, predict option-level marginal costs and travel-time proxies, solve the pricing problem in Eq. (eq:pricingpred), and display the resulting menu to the passenger.

# 4.2.1. State encoding and training data generation

The first step in the pricing pipeline is to convert the dynamic state of the DRT system into a format suitable for machine learning. We achieve this through a novel spatiotemporal encoding scheme. The state $s _ { t }$ encoded through a dual-channel architecture denoted $\phi ( s _ { t } , k )$ , which integrates both spatial and temporal dimensions. Spatially, the total service area is partitioned into discrete spatial cells, and the booking horizon is divided into $T$ predefined equal-length time intervals. Passengers are then classified into these spatiotemporal bins based on their arrival time and pickup location, and their counts are aggregated. This encoding process is illustrated in Fig. 5, where boarding locations are represented as dots, and the corresponding arrival time interval for each passenger is indicated by the number within the dot.

While the precise daily passenger volume remains unknown, the spatiotemporal distribution of arrivals—as captured by the above encoding—can be predicted using CNN-based models. Both the spatial discretization granularity and the number of time intervals are tunable parameters in this framework.

![](images/2b96730de49874e0b7c800b18db3bd76be356e64ce4092430eda266e98b0ad62.jpg)

Fig. 5. State encoding

To train the CNN, we construct the labels by utilizing a marginal insertion cost computation method. Specifically, for each passenger arrival and each candidate option $k \in O _ { t }$ , we first remove booking $( t , k )$ from the final route $R ^ { T }$ , then re-optimized terminal route problem to obtain $R _ { - t k } ^ { T }$ , and finally define the cost difference between $R ^ { T }$ and $R _ { - t k } ^ { T }$ as the true marginal insertion cost $c _ { t k }$ . As a simplified example, suppose there are three passengers. In this case, we solve four VRP instances: one baseline instance containing all three passengers, and three modified VRP instances, each excluding one of Passengers 1, 2, or 3 individually. This yields three marginal insertion costs, defined as the cost differences between the baseline solution and each modified instance. The training procedure evaluates all candidate options for each simulated arrival, rather than only the option chosen by the passenger. With more simulated booking days, the dataset covers a broader range of likely state–option pairs under the demand model and operating policy. Although the full DRT x , ’ ability to learn option-level cost and travel-time differences in frequently observed conditions. The data generation procedure during simulation is depicted in Fig. 6.

![](images/b7d491e5dcc3c2f3da9a62656ba725a45c97e74001255021ea5d978450681174.jpg)  
(a) All passengers

![](images/7b89592b46a73e9cf4197c4fa83c0fa76a59e03faecccff70cf5ea36f5310872.jpg)  
(b) Excluding pas senger 1

![](images/8673b4ab8f82b9d115a1affe8f6d3c322a40e6fa000b5c5a2a1bbc973bb014f8.jpg)  
(c) Excluding pass enger 2

![](images/d4d40bf88d4e0153f33ff454e3d7b9a540501a9ac691ae63e4e63116595df92a.jpg)  
(d) Excluding passenger 3   
Fig. 6. Training data generation process

# 4.2.2. Cost and in-vehicle travel time prediction module

When a new passenger arrives, the system immediately predicts two vectors to support real-time decision-making: i) the marginal insertion cost of each candidate boarding location, which determines our

dynamic pricing; and ii) the predicted service position of each boarding location, denoted $q _ { t k }$ , which is used to display $\tau _ { t k }$ .

To accomplish this dual-output predicting task, we employ a basic CNN architecture consisting of two convolutional layers, one average pooling layer, and two fully connected layers. We choose this structure because it is particularly good at capturing hierarchical spatiotemporal information from the encoded system state $\phi ( s _ { t } , k )$ . This model is trained with historical data, the input feature is state encoding, and the target label is the pre-calculated marginal insertion cost. The goal of our module is to accurately predict the true but unknown costs $c _ { t k }$ of inserting a location $k$ for a passenger that arrived at time $t$ in the final route $R ^ { T }$ .

The network processes the encoded input state $\phi ( s _ { t } , k )$ through the shared convolution layer and pooling layer to extract important features. These features are then fed into two separate output branches, one generates the predicted marginal insertion cost $\hat { c } _ { t k }$ for each boarding location, and the other predicts the corresponding service position $\widehat { q } _ { t k }$ for subsequent in-vehicle travel time calculation. The core cost prediction formula is given by

$$
\left(\hat {c} _ {t k}, \hat {q} _ {t k}\right) = \mathcal {N} _ {\theta} \left(\phi \left(s _ {t}, k\right)\right), \tag {12}
$$

where $\mathcal { N } _ { \theta }$ represents a network with trainable parameters $\theta$ , and the state encoding $\phi ( s _ { t } , k )$ contains the distribution of passengers in time and space. This predicted cost $\hat { c } _ { t k }$ is an estimate of the invisible real cost $c _ { t k }$ , and it will be directly sent to the following pricing optimization module (Section 4.2.3) for pricing.

The in-vehicle travel time is crucial for enhancing passenger experience. Previous research has long shown that passengers attach great importance to the certainty and predictability of travel time (Aflaki & Zhang, 2025; Yan et al., 2022). The second output of the network is the predicted service position $\widehat { q } _ { t k }$ . We then use $\widehat { q } _ { t k }$ to calculate the expected in-vehicle travel time. Specifically, First, a data-driven method predicts the service sequence for each boarding location based on historical operating data, drawing lessons from the spatial distribution of boarding points and historical VRP data. Second, based on the predicted service position, we combine preset speed and distance parameters to calculate the expected invehicle travel time. In the end, the system simultaneously presents the price and predicted in-vehicle travel time for each recommended boarding option, allowing passengers to make an informed choice.

# 4.2.3. Pricing optimization based on predicted costs

The predicted marginal insertion cost is then used to inform real-time pricing strategies. As detailed in Section 3.3, we delve into a dynamic-programming approach, concentrating on each incoming passenger

and tackling a problem of maximizing expected profit per passenger. Once the optimal pricing options have been determined and the potential costs have been estimated, the operator faces a pressing real-time pricing challenge for passengers. The economic principle is simple: each boarding choice yields fare income, but it also adds an operational strain to the system. DRPO estimates this strain by considering the marginal insertion cost of each option and then sets prices to maximize expected profit.

In order to delve into the pricing issue, we first establish the optimal pricing strategy based on the actual marginal insertion cost. Take, for instance, a passenger arriving at epoch $t$ , faced with the option set, labeled as $O _ { t }$ . The price-adjustment vector for these options is denoted as $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ , where each element $a _ { t k }$ corresponds to a choice within $O _ { t }$ . Consequently, the operator's objective to maximize expected profit per passenger can be formulated as follows:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - c _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right), \tag {13}
$$

where $P _ { t k } ( \pmb { a } _ { t } )$ is the probability that a passenger selects option $k$ . $( f _ { e } + a _ { t k } - c _ { t k } )$ is the realized profit if option $k$ is chosen.

The practical challenge here is that $c _ { t k }$ remains out of sight when it's time to set prices, since this value hinges on future customer arrivals and the ultimate routing strategy. To solve this hurdle, DRPO swaps out $c _ { t k }$ with the predicted marginal insertion cost $\hat { c } _ { t k }$ generated by the neural network, which transforms the theoretical model into a practical online problem:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - \hat {c} _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right). \tag {14}
$$

To streamline the process, we eliminate the time and segment indices and concentrate on a single passenger. Let $O$ represent the range of available choices, let $a _ { k }$ be the price adjustment for choice $k \in O$ , and let $\pmb { a } = ( a _ { k } ) _ { k \in O }$ be the price vector. Similarly, let $c _ { k }$ and $\hat { c } _ { k }$ denote the actual and predicted marginal insertion costs, expressed as vectors $\pmb { c } = ( c _ { k } ) _ { k \in O }$ and $\hat { \pmb { c } } = ( \hat { c } _ { k } ) _ { k \in O }$ . Let $P _ { k } ( { \pmb a } )$ represent the probability that choice $k$ will be chosen. The true-cost and prediction-based objectives can then be expressed as

$$
R (\boldsymbol {a}; \boldsymbol {c}) = \sum_ {k \in O} \left(f _ {e} + a _ {k} - c _ {k}\right) P _ {k} (\boldsymbol {a}), \tag {15}
$$

and

$$
R (\boldsymbol {a}; \hat {\boldsymbol {c}}) = \sum_ {k \in O} \left(f _ {e} + a _ {k} - \hat {c} _ {k}\right) P _ {k} (\boldsymbol {a}), \tag {16}
$$

so that the ideal and practical pricing problems are therefore

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}), \quad \max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \hat {\boldsymbol {c}}), \tag {17}
$$

where $\mathcal { A }$ denotes the feasible price set. To avoid overly aggressive pricing, we set operational restrictions on discounts and surcharges. Appendix C derives the unconstrained structural characterization of the optimizer and shows how the optimal prices depend on the cost vector. In the implementation, however, the online policy solves the constrained pricing problem over $\mathcal { A }$ numerically, using the Appendix C characterization as a structural guide. For any cost vector $\pmb { c }$ , let

$$
\boldsymbol {a} ^ {*} (\boldsymbol {c}) \in \underset {\boldsymbol {a} \in \mathcal {A}} {\operatorname {a r g m a x}} R (\boldsymbol {a}; \boldsymbol {c}) \tag {18}
$$

denote the optimal pricing decision. Likewise, for a predicted cost vector ??̂, let

$$
\boldsymbol {a} ^ {*} (\hat {\boldsymbol {c}}) \in \underset {\boldsymbol {a} \in \mathcal {A}} {\operatorname {a r g m a x}} R (\boldsymbol {a}; \hat {\boldsymbol {c}}). \tag {19}
$$

The mapping $\pmb { c } \mapsto \pmb { a } ^ { * } ( \pmb { c } )$ serves as the downstream pricing oracle in DRPO. It can be handled analytically within the MNL framework. The crux of the matter is that the worth of predictions should be gauged based on the effectiveness of the induced pricing decision, rather than just the pointwise accuracy of ??̂.

# 4.2.4. SPO decision-focused learning objective

The prediction module isn't just a standalone goal; it's a stepping stone to inform our pricing strategies. Just because we aim to minimize prediction loss on the cost margin, it doesn't automatically guarantee smarter pricing choices. Tiny forecasting errors can still lead to subpar price sets, potentially resulting in a significant drop in our profit margins. To bridge the gap between our predictions and actual decisions, we're leaning on Elmachtoub and Grigas's (2022) Smart Predict-Then-Optimize (SPO) framework. We're fine-tuning our model using a loss function that focus on the quality of the resulting pricing decisions.

In order to fit the SPO framework, we start by rephrasing the pricing issue discussed in Section 4.2.3 as a linear-objective canonical form, while absorbing the non-linear relationship with prices into the lifted decision vector ${ \pmb w } ( { \pmb a } )$ . For a fixed passenger, let $\pmb { c } = ( c _ { k } ) _ { k \in O }$ and $\hat { \pmb { c } } = ( \hat { c } _ { k } ) _ { k \in O }$ denote the true and predicted marginal-cost vectors. The downstream pricing problem is:

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}). \tag {20}
$$

We define

$$
R _ {0} (\boldsymbol {a}) = \sum_ {k \in O} \left(f _ {e} + a _ {k}\right) P _ {k} (\boldsymbol {a}), \quad \boldsymbol {P} (\boldsymbol {a}) = \left[ P _ {k} (\boldsymbol {a}) \right] _ {k \in O}, \tag {21}
$$

so that

$$
R (\boldsymbol {a}; \boldsymbol {c}) = R _ {0} (\boldsymbol {a}) - \boldsymbol {c} ^ {\top} \boldsymbol {P} (\boldsymbol {a}). \tag {22}
$$

We then introduce the augmented cost vectors and lifted decision vector

$$
\boldsymbol {c} _ {a u g} = \left[ \begin{array}{l} \boldsymbol {c} \\ 1 \end{array} \right], \quad \hat {\boldsymbol {c}} _ {a u g} = \left[ \begin{array}{l} \hat {\boldsymbol {c}} \\ 1 \end{array} \right], \quad \boldsymbol {w} (\boldsymbol {a}) = \left[ \begin{array}{l} \boldsymbol {P} (\boldsymbol {a}) \\ - R _ {0} (\boldsymbol {a}) \end{array} \right]. \tag {23}
$$

For clarity, let

$$
\boldsymbol {y} = \boldsymbol {c} _ {a u g}, \quad \hat {\boldsymbol {y}} = \hat {\boldsymbol {c}} _ {a u g}. \tag {24}
$$

Then, for any feasible $\pmb { a } \in \mathcal { A }$

$$
\boldsymbol {y} ^ {\top} \boldsymbol {w} (\boldsymbol {a}) = \boldsymbol {c} ^ {\top} \boldsymbol {P} (\boldsymbol {a}) - R _ {0} (\boldsymbol {a}) = - R (\boldsymbol {a}; \boldsymbol {c}). \tag {25}
$$

Hence, maximizing downstream profit is equivalent to minimizing a linear objective:

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}) \Leftrightarrow \min  _ {\boldsymbol {a} \in \mathcal {A}} \boldsymbol {y} ^ {\top} \boldsymbol {w} (\boldsymbol {a}). \tag {26}
$$

Considering the set $\mathscr { W } = c o n v \pmb { w } ( \pmb { a } ) ; \pmb { a } \in \mathcal { A }$ , defined as the convex hull of all feasible decision. Given that our objective is linear, minimizing it across $\mathcal { W }$ guarantees that the optimal value remains unchanged. Furthermore, $\mathcal { W }$ is not just nonempty but also has well-defined bounds. This is due to the fact that $\pmb { P } ( \pmb { a } )$ resides within the probability simplex, and when prices are bounded, so too is the function $R _ { 0 } ( \mathbf { a } )$ . As a result, the pricing problem neatly fits within the realm of the standard SPO framework. We define

$$
z ^ {*} (\boldsymbol {y}) = \underset {\boldsymbol {w} \in \mathcal {W}} {\operatorname {m i n}} \boldsymbol {y} ^ {\top} \boldsymbol {w}, \quad \boldsymbol {w} ^ {*} (\boldsymbol {y}) \in \operatorname {a r g m i n} _ {\boldsymbol {w} \in \mathcal {W}} \boldsymbol {y} ^ {\top} \boldsymbol {w}. \tag {27}
$$

Here, $z ^ { \ast } ( y )$ denotes the optimal objective value under cost vector ??, while $\pmb { w } ^ { * } ( \pmb { y } )$ denotes an optimal lifted decision achieving that value. In our setting, the oracle executes by solving the original pricing problem defined in Section 4.2.3 and then translate the resulting optimal price vector into its lifted representation.

The SPO loss quantifies the downstream regret associated with pricing strategies that rely on predicted rather than actual cost vectors.

$$
\ell_ {S P O} (\hat {\mathbf {y}}, \mathbf {y}) = \mathbf {y} ^ {\top} \mathbf {w} ^ {*} (\hat {\mathbf {y}}) - \mathbf {z} ^ {*} (\mathbf {y}). \tag {28}
$$

Using Eq. (26), this can be written directly as the profit loss induced by prediction error:

$$
\ell_ {S P O} (\hat {\mathbf {y}}, \mathbf {y}) = R \left(\mathbf {a} ^ {*} (\mathbf {c}); \mathbf {c}\right) - R \left(\mathbf {a} ^ {*} (\hat {\mathbf {c}}); \mathbf {c}\right). \tag {29}
$$

Given that $\ell _ { S P O }$ typically presents challenges due to its nonconvex nature and lack of differentiability with respect to predictions, we turn to the convex surrogate loss $\ell _ { S P O + }$ introduced by Elmachtoub and Grigas (2022). Within our framework, this alternative formulation takes the following expression:

$$
\ell_ {S P O +} (\hat {\mathbf {y}}, \mathbf {y}) = \max  _ {\mathbf {w} \in \mathcal {W}} (\mathbf {y} - 2 \hat {\mathbf {y}}) ^ {\top} \mathbf {w} + 2 \hat {\mathbf {y}} ^ {\top} \mathbf {w} ^ {*} (\mathbf {y}) - z ^ {*} (\mathbf {y}). \tag {30}
$$

A valid subgradient relative to the augmented prediction is given by:

$$
2 \left(\boldsymbol {w} ^ {*} (\boldsymbol {y}) - \boldsymbol {w} ^ {*} (2 \hat {\boldsymbol {y}} - \boldsymbol {y})\right) \in \partial_ {\hat {\boldsymbol {y}}} \ell_ {S P O +} (\hat {\boldsymbol {y}}, \boldsymbol {y}). \tag {31}
$$

Given that the final element of $\widehat { \pmb { y } }$ is set to 1, the backpropagation process is limited to the initial $| O |$ components associated with ??̂. Despite the system running in a sequential, real-time manner, the actual model training is conducted off-line. During each decision epoch $t$ , the neural network crunches the encoded state $\phi ( s _ { t } , k )$ to predict the option-specific marginal insertion cost vector $\hat { \pmb { c } }$ , which is then sent to the pricing oracle to derive ${ \pmb a } ^ { * } ( \hat { \pmb c } )$ . Once the episode wraps up, we reconstruct the actual cost vector ?? from the final-route data, pair it with $( \widehat { \pmb { y } } , \pmb { y } )$ , and backpropagate through the prediction network.

To enhance training stability, we integrate $\mathrm { S P O + }$ loss with Huber loss for marginal insertion cost prediction. The comprehensive training objective is

$$
\mathcal {L} (\theta) = \alpha_ {S P O +} \ell_ {S P O +} (\widehat {\boldsymbol {y}}, \boldsymbol {y}) + (1 - \alpha_ {S P O +}) \mathcal {L} _ {H u b e r} (\widehat {\boldsymbol {c}}, \boldsymbol {c}). \tag {32}
$$

The hyperparameter $\alpha _ { S P O + } \in [ 0 , 1 ]$ strikes a balance between optimizing for final decisions and achieving point-level prediction accuracy. Initially, the model undergoes an initial warm-up period where it's trained exclusively with the Huber loss, enabling it to first build robust spatial and cost representations. After this foundation is set, we gradually dial up $\alpha _ { S P O + }$ , causing the optimization objective to gradually pivot toward enhancing downstream decision performance. This hybrid training approach not only boosts optimization stability but also maintains the decision-centric philosophy inherent in the DRPO framework.

# 5. Results

This section evaluates the performance of the DRPO framework and compares it with several benchmark strategies.

# 5.1. Experimental design

We assess DRPO in a many-to-one DRT framework where passengers start from various locations and head to a single destination. Each strategy is evaluated under identical default conditions and passenger decision-making parameters, including the option to opt out, ensuring that passenger departure remains influenced by the pricing strategy. As a result, Section 5.2 and Section 5.3 present both provider-side metrics, like overall cost and net earnings, as well as passenger-side results, such as the proportion of

home pickup and the quit rate. In Section 5.4, we explore how these outcomes change under different behavioral assumptions. We also contrast DRPO with five standard strategies:

Only-home: all passengers select home pickup.

Only-meeting-points: all passengers choose recommended meeting points.

No-pricing: passengers can choose either home pickup or a recommended meeting point for the same price.

Static-pricing: provides fixed discounts for recommended meeting points and fixed extra charges for home pickup.

DSPO: a learning-based pricing benchmark proposed by Akkerman et al. (2024).

DRPO: our decision-focused learning pricing framework.

The Only-home and Only-meeting-points scenarios represent the upper and lower bounds on operational costs, respectively. The range between these bounds quantifies the potential room for cost improvement. The No-pricing baseline reveals passengers' intrinsic preferences for boarding locations in the absence of incentives. In contrast, the Static-pricing baseline demonstrates the benefits of introducing a simple pricing strategy. To evaluate the effectiveness of our method, we compared it against the DSPO framework. We utilize a small experiment to study the basic problems in Section 5.2, and then show the results of a semi-realistic Beijing case in Section 5.3.

# 5.2. Synthetic case

We utilize a synthetic case study to show the basic properties of the DRPO framework.

# 5.2.1. Problem setting

Using the Gehring and Homberger (2002) benchmark, we generated synthetic demand data comprising 200 geographic locations. These locations were randomly partitioned into mutually exclusive training and test sets, each containing 100 locations. From each set, we randomly sampled 10 locations as candidate recommended meeting points. As shown in Fig. 7.

Driving times are calculated based on Euclidean distances. We established a vehicle velocity of 30 kilometer per hour, driver wage at $\yen 30$ hourly, fuel cost at $\yen 0.6$ per distance unit.

![](images/107b07f7a4ca6a4d5356905ff2dbd2fa9cc101f011ea568abc7b440b18ade50b.jpg)  
Fig. 7. Spatial distribution of home locations and meeting points in synthetic case.

In our experimental setup, dwell time is modeled as spatially heterogeneous variables, with their distribution parameters derived from the six-hump camel-back function, which is a standard multimodal test function in optimization literature.

This closely resembles reality as dwell time in, for instance, high-density residential areas are expected to be higher compared with rural areas or suburban stops. Dwell time at location $i$ are obtained by projecting the service area onto the domain $( x \in [ - 3 , 3 ] , y \in [ - 2 , 2 ] )$ of

$$
l _ {i} (x, y) = \left(4 - 2. 1 x ^ {2} + \frac {x ^ {4}}{3}\right) x ^ {2} + x y + (- 4 + 4 y ^ {2}) y ^ {2}, \tag {33}
$$

which is an optimization function proposed in Molga and Smutnicki (2005). We bound the dwell time on $l _ { i } \in [ 1 , 1 0 ]$ minutes.

The number of arriving passengers on a day from the negative binomial distribution $D$ , where the probability mass function is given by

$$
P r (D = d) = (\frac {d + r - 1}{d}) (1 - p) ^ {d} p ^ {r}, \quad d = 0, 1, 2, \dots , \tag {34}
$$

where $r = 9 0$ , $p = 0 . 5$ , and $\begin{array} { r } { \mathbb { E } [ D ] = \frac { r ( 1 - p ) } { p } = 9 0 ~ } \end{array}$ . The passenger arrival times follow a uniform distribution throughout the service period. The daily maximum passenger capacity is constrained by our bus fleet size: we operate a limited fleet of 12 buses, each with a maximum seating capacity of 12 passengers per bus. As actual passenger choice data is not available, passenger decisions are modeled using the MNL model outlined in Section 3.4.

# 5.2.2. Analysis of results

We set the max number of episodes to 200 in all strategies. Table 1 reports the corresponding averages over 30 random seeds.

Table 1. Results on the synthetic case.   

<table><tr><td>Strategy</td><td>Home-pickup share</td><td>Quit rate</td><td>Total costs</td><td>Cost saving vs. Only-home</td><td>Net profit</td></tr><tr><td>Only-home</td><td>99.00%</td><td>1.01%</td><td>2890.75</td><td>0.00%</td><td>1590.92</td></tr><tr><td>Only-meeting - points</td><td>0.00%</td><td>45.23%</td><td>702.95</td><td>75.68%</td><td>1780.39</td></tr><tr><td>No-pricing</td><td>98.00%</td><td>1.01%</td><td>2885.99</td><td>0.16%</td><td>1595.67</td></tr><tr><td>Static-pricing</td><td>22.00%</td><td>1.20%</td><td>2382.40</td><td>17.59%</td><td>2090.93</td></tr><tr><td>DSPO</td><td>53.66%</td><td>3.69%</td><td>2218.46</td><td>23.26%</td><td>2193.79</td></tr><tr><td>DRPO</td><td>45.32%</td><td>3.11%</td><td>2093.06</td><td>27.59%</td><td>2346.10</td></tr></table>

Table 1 shows a straightforward comparison of the benchmark policies. The Only-home and No-pricing options yield almost identical results, suggesting that, in the absence of clear incentives, most travelers

still opt for home pickup. On the other end of the spectrum, Only-meeting-points has the lowest overall cost, but this strategy comes with a significant $4 5 . 2 3 \%$ dropout rate, rendering it less operational than its efficiency-focused counterpart. Static pricing already significantly outperforms these basic models, cutting total costs by $1 7 . 5 9 \%$ compared to only-home and boosting net profits to an impressive 2090.93.

Among the practical pricing strategies, DRPO exhibits the best overall performance. Its average total cost comes in at 2093.06, which shakes out to a $2 7 . 5 9 \%$ drop compared to Only-home and a $1 2 . 1 4 \%$ decrease relative to Static-pricing. Meanwhile, its mean net profit hits 2346.10, blowing the competition out of the water. Compared to DSPO, DRPO boosts net profit by $6 . 9 4 \%$ and cuts total costs by $5 . 6 5 \%$ , while also dialing down the home-pickup share from $5 3 . 6 6 \%$ to $4 5 . 3 2 \%$ . In the paired 30-seed comparison, achieves a higher net profit than DSPO in 29 out of 30 cases. All told, these findings indicate that decisionfocused end-to-end training allows the operator to direct more passengers to meeting points, while avoiding the unrealistic, high-quit extreme that is characteristic of a pure meeting-point policy.

# 5.3. Yanjiao to Guomao case in Beijing

We conduct a semi-realistic case study on the Yanjiao-Guomao commuter corridor in Beijing to evaluate the practical applicability of the DRPO framework.

# 5.3.1. Problem setting

In order to assess the practical applicability of our DRPO framework, we conduct a semi-realistic case study in Beijing. Specifically, we focus on the commuter route from the suburbs outside the Fifth Ring Road in Beijing to the Central business district (CBD). We utilize the bus stop data published by Beijing Municipal Transportation Commission and select 170 existing bus stops as recommended meeting points.

Passenger demand is generated with parameters $r = 2 4 0$ and $p = 0 . 5$ , so that $\begin{array} { r } { \mathbb { E } [ D ] = { \frac { r ( 1 - p ) } { p } } = 2 4 0 } \end{array}$ . Then, we randomly generate 300 home locations, sampling them according to population density derived from official statistics for Beijing.

There are 25 buses in our system, each with a capacity of 12 seats. The fare structure includes a base fare $f _ { e }$ of 30 units, with dynamic pricing adjustments ranging from [−10, 10] units, as determined by the DRPO framework. As in the synthetic case, we set the following parameters: vehicle speed $= 3 0 \ \mathrm { k m / h }$ , driver wage $= \yen 30/ h$ , fuel cost $= \yen 0.6/\mathrm { k m }$ . Fig.8 shows the spatial distribution.

![](images/0d37432644f208c903822dc13d14df529487c4b826e0bc6d571cffd45cf47234.jpg)  
Fig. 8. Spatial distribution of home locations and meeting points in Beijing case

# 5.3.2. Analysis of results

Table 2 reveals the performance of the DRPO framework in comparison with the other six strategies in a semi-realistic Beijing scenario.

Table 2. Results on the Beijing case.   

<table><tr><td>Strategy</td><td>Home-pickup share</td><td>Quit rate</td><td>Total costs</td><td>Cost saving vs. Only-home</td><td>Net profit</td></tr><tr><td>Only-home</td><td>99.26%</td><td>0.74%</td><td>5115.22</td><td>0.00%</td><td>6644.78</td></tr><tr><td>Only-meeting - points</td><td>0.00%</td><td>52.63%</td><td>337.30</td><td>93.41%</td><td>5272.70</td></tr><tr><td>No-pricing</td><td>98.18%</td><td>0.74%</td><td>5093.67</td><td>0.42%</td><td>6666.33</td></tr><tr><td>Static-pricing</td><td>72.94%</td><td>2.03%</td><td>4298.90</td><td>15.96%</td><td>7311.10</td></tr><tr><td rowspan="2">DSPO</td><td>45.35</td><td>4.04%</td><td>3717.2</td><td>27.33%</td><td>7656.22</td></tr><tr><td>%</td><td></td><td>8</td><td></td><td></td></tr><tr><td rowspan="2">DRPO</td><td>37.18</td><td>3.24%</td><td>3667.1</td><td>28.31%</td><td>7798.82</td></tr><tr><td>%</td><td></td><td>8</td><td></td><td></td></tr></table>

Table 2 showcases a straightforward comparison of the benchmark policies. Only-home and No-pricing yield nearly identical results, suggesting that, in the absence of clear incentives, most passengers still opt for home pickup. On the opposite, Only-meeting-points results in the lowest overall expense, yet it also boasts a staggering $5 2 . 6 3 \%$ dropout rate and the smallest net income among the feasible pricing strategies, rendering it unappealing from an operational standpoint despite its efficient routing. While Static-pricing offers a slight enhancement over these simplistic approaches, it still lags significantly behind the learningbased methodologies.

In the realm of practical pricing strategies, DRPO shines brightest in the semi-realistic Beijing scenario. Its average total cost stands at 3667.18, marking a remarkable $2 8 . 3 1 \%$ decrease compared to Only-home and a modest $1 . 3 5 \%$ reduction against DSPO. Meanwhile, its average net profit hits a high of 7798.82, outperforming all other tested methods. DRPO bests DSPO by a significant $1 . 8 6 \%$ in net profit, boosts the home-pickup share from $4 5 . 3 5 \%$ to $3 7 . 1 8 \%$ , and cuts the quit rate from $4 . 0 4 \%$ to $3 . 2 4 \%$ . These findings suggest that decision-focused training can guide more passengers to designated meeting points.

Overall, the Beijing case confirms that DRPO can be effectively applied in large-scale, real-world scenarios. It consistently outperforms the other strategies in profit improvement.

# 5.4. Sensitivity analysis

In our quest to assess the resilience of the DRPO system, we delve into a series of sensitivity analyses on the synthetic benchmark. The objective is to illustrate the impact of various behavioral factors in the passenger choice model on system efficiency and to pinpoint the circumstances under which dynamic pricing yields optimal results. More precisely, we modify the price-sensitivity parameter $\beta ^ { p r i c e }$ , the outside-option utility $U _ { 0 }$ , and the home-pickup alternative-specific constant $A S C _ { h o m e }$ , which reflects passengers' natural inclination towards home pickup.

# 5.4.1. Experimental settings

We conduct sensitivity analyses on the synthetic benchmark, tweaking one variable at a time while keeping everything else constant. Key performance indicators include net profit, total operating cost, and

quit rate. These figures collectively reflect both the operational efficiency of the operator and the passenger response.

To capture passengers' natural inclination toward home pickup, we incorporate $A S C _ { h o m e }$ as a constant specific to this option. A higher $A S C _ { h o m e }$ value points to a stronger preference for door-to-door service, while a lower value suggests passengers are more open to walking to a suggested meeting point. Consequently, the utility function for the home-pickup option is therefore modified as follows:

$$
U _ {t h _ {t} g} = A S C _ {h o m e} - \beta_ {g} ^ {i v t} \tau_ {t h _ {t}} - \beta_ {g} ^ {p r i c e} a _ {t h _ {t}} + \varepsilon_ {t h _ {t}}, \tag {35}
$$

while the utility of meeting-point options remains unchanged.

For each variable, we assess it across five distinct levels, whereas the other behavioral variables remain at their standard settings $( \beta ^ { p r i c e } = 0 . 2 5$ , $U _ { 0 } = - 1 . 0$ , and $A S C _ { h o m e } = 1 . 4 )$ ). Note that the price coefficient within the utility function is $- \beta ^ { p r i c e } = - 0 . 2 5$ . Each test involves DRPO, with three repetitions for each level.

# 5.4.2. Effect of price sensitivity

We first examine the effect of the price-sensitivity parameter $\beta ^ { p r i c e }$ . This parameter dictates the extent to which passengers react to discounts on meeting points and surcharges for home pickup. Given that the utility coefficient for price sits at $- \beta ^ { p r i c e }$ , a higher value of $\beta ^ { p r i c e }$ translates to a more pronounced negative price coefficient. This means passengers become more sensitive to price changes, handing the operator greater leverage in steering where passenger decide to board.

Fig. 9 illustrates that increasing $\beta ^ { p r i c e }$ from 0.15 to 0.35, results in a consistent enhancement in operational performance. Specifically, the net profit surges by $6 4 . 7 \%$ , while the total operating cost decreases by $4 3 . 6 \%$ . This observation underscores the critical role of price elasticity in determining the effectiveness of DRPO. In scenarios where passengers are highly attuned to financial incentives, operators have the potential to reallocate more demand away from home pickup to meeting points, thereby minimizing unnecessary travel and enhancing overall routing efficiency.

![](images/64c8b089843070865ea7549999b235c2079d100480c690522ca8c87ed2b89942.jpg)  
Fig. 9. Price-sensitivity parameter $\beta ^ { p r i c e }$ . The top figure reports relative changes in mean net profit and mean total cost compared with the default setting $\beta ^ { p r i c e } = 0 . 2 5$ , and the bottom figure reports the mean quit rate

In contrast, if passengers do not care much about the cost, the same pricing strategy does not have much of an impact on how they board. In these situations, the benefits from dynamic pricing are usually smaller. Thus, price-sensitive is key to understanding the worth of dynamic pricing in this situation.

# 5.4.3. Effect of outside option utility

We next consider the outside-option utility $V _ { 0 }$ , which represents the attractiveness of not using the DRT service. A higher value of $V _ { 0 }$ suggests that opting for alternative transportation methods, like personal vehicles or regular public transit, becomes more attractive compared to the DRT service.

Fig. 10 illustrates that DRPO is incredibly responsive to fluctuations in $V _ { 0 }$ . When $V _ { 0 } \leq - 1 . 0$ , the DRT service maintains a competitive edge, offering a robust framework with minimal quit rates. But as $V _ { 0 }$ hits zero and climbs, the quit rate spikes. For instance, it jumps from $4 . 0 \%$ at $V _ { 0 } = - 1 . 0$ to a stunning $6 0 . 5 \%$ at $V _ { 0 } = 0 . 0$ , and a sky-high $9 5 . 2 \%$ at $V _ { 0 } = 1 . 0$ . Concurrently, total operational costs drop quickly, plummeting to a mere 286.3 at $V _ { 0 } = 1 . 0$ .

![](images/34016b754808792369b3af160701f8f52642c175d2e8d77f8038d4b90589597e.jpg)  
Fig. 10. Outside-option level $V _ { 0 }$ . The top figure reports relative changes in mean net profit and mean total cost compared with the default setting $V _ { 0 } = - 1 . 0$ , and the bottom figure reports the mean quit rate

The apparent cost savings need to be interpreted closely. In competitive environment, the decline in cost is not due to better routing efficiency, but rather the loss of demand. Therefore, low operational costs

should not be considered in isolation when outside options are particularly appealing. They should be assessed in conjunction with demand-side metrics like the quit rate or passenger retention.

# 5.4.4. Effect of home-pickup preference

Finally, we examine the role of the home-pickup alternative-specific constant $A S C _ { h o m e }$ , which captures intrinsic preference for home pickup. When the value of $A S C _ { h o m e }$ is lower, it essentially signals that passengers are less wedded to the convenience of a home pickup, making them more amenable to choose meeting points.

Fig. 11 shows that lower values of $A S C _ { h o m e }$ improve operator-side efficiency. As $A S C _ { h o m e }$ drops from 1.8 to 1.0, the net profit increases by $3 7 . 2 \%$ while the total operating cost declines by $4 1 . 3 \%$ . This trend suggests that when passengers are less reliant on home pickup, DRPO can more efficiently redirect demand towards meeting point, thereby decreasing the routing strain.

![](images/a869e96d4bb21251467168c2f03fb704d93e5f4e4350beaf74683b647ffa5115.jpg)

Fig. 11. Home-pickup preference parameter $A S C _ { h o m e }$ . The top figure reports relative changes in mean net profit and mean total cost compared with the default setting $A S C _ { h o m e } = 1 . 4$ , and the bottom figure reports the mean quit rate

Concurrently, the findings reveal an important trade-off. As the $A S C _ { h o m e }$ decreases, there is a notable uptick in passenger quit rates. This escalates from a mere $1 . 7 \%$ when $A S C _ { h o m e } = 1 . 8$ to a substantial $8 . 1 \%$ at $A S C _ { h o m e } = 1 . 0$ . So, while a weaker inclination towards home pickup allows for more room for efficiency gains, but also could potentially turn off some passengers to the service entirely.

# 5.4.5. Summary of sensitivity analysis

In essence, the sensitivity analysis serves as a complement to the primary experiments, demonstrating that DRPO retains its performance edge across various behavioral scenarios. Furthermore, the findings suggest that any improvements made on the operator's side should not be viewed in isolation, as costsaving measures might lead to decreased passenger retention if the alternative options are appealing or if service quality is diminished.

# 6. Conclusion

In this study, we delve into a dynamic decision problem that integrates meeting-point recommendations with pricing strategies in many-to-one DRT systems. Unlike previous research, which often assumes uniform boarding patterns or concentrates on routing efficiency, we meticulously simulate the sequence of passenger arrivals, diverse passenger behavior, and the delicate balance between personalized convenience and overall system performance.

We tackle this challenge by framing it as a Markov decision process where the system first generates potential boarding choices before dishing out tailored price incentives to each new passenger. To solve this problem, we introduce DRPO, a decision-focused framework that marries heuristic meeting-point suggestions with SPO-based learning framework, CNN-driven state encoding, and marginal insertion cost forecasting.

The results of our experiments on a synthetic benchmark and a semi-realistic Beijing case study illustrate about the efficacy of our proposed approach. In the synthetic scenario, DRPO not only tops the charts in terms of mean net profit but also cuts costs by a significant $2 7 . 6 \%$ compared to Only-home. In the Beijing case, DRPO once again delivers superior results, achieving the highest mean net profit among

the strategies tested. It also reduces costs by an impressive $2 8 . 3 1 \%$ relative to Only-home, and boosts net profit by a modest yet noteworthy $1 . 8 6 \%$ over DSPO. The sensitivity analysis reveals that these benefits are most pronounced when passengers are incentivized by price, and it underscores the importance of considering joint interpretations of cost reductions alongside quit behaviors as external options become more appealing.

The results reveal three key insights. Firstly, dynamic incentives can successfully guide a considerable number of passengers from home pickup to designated meeting points without pushing the system towards an unrealistic high-exit policy. Secondly, explicit behavioral modeling is crucial for assessing both operational efficiency and passenger retention. Lastly, decision-focused learning enhances the coordination between cost prediction and subsequent pricing strategies, ultimately resulting in superior performance compared to other models.

We present a scalable, data-driven framework for integrating meeting-point recommendation and dynamic pricing in DRT systems. It offers a method for operators to balance efficiency, profitability, and passenger acceptance under unknown demand and operational constraints. This framework can be extended in future work by designing meeting points, integrate many-to-many scenarios, and incorporating more constraints (e.g., time windows) or network features (e.g., feeder services).

Appendix A: Notation   

<table><tr><td>Variable</td><td>Description</td></tr><tr><td>Sets and indices</td><td></td></tr><tr><td>G</td><td>Set of passenger segments.</td></tr><tr><td>g</td><td>Passenger-segment index.</td></tr><tr><td>t</td><td>Decision epoch or request index.</td></tr><tr><td>Mt</td><td>Set of recommended meeting point options for passenger t.</td></tr><tr><td>Ot</td><td>Full set of boarding options offered to passenger t, given by O_t = {ht} ∪ Mt.</td></tr><tr><td>k</td><td>Boarding-option index in O_t.</td></tr><tr><td>ht</td><td>Home-pickup option for passenger t.</td></tr><tr><td>0</td><td>Outside-option index.</td></tr><tr><td>Cost, revenue, and routing variables</td><td></td></tr><tr><td>fe</td><td>Base fare.</td></tr><tr><td>at = (atk)k∈Ot</td><td>Price-adjustment vector for passenger t.</td></tr><tr><td>atk</td><td>Price adjustment for option k of passenger t.</td></tr><tr><td>ctk</td><td>True marginal insertion cost.</td></tr><tr><td>ˆtk</td><td>Predicted marginal insertion cost.</td></tr><tr><td>cw</td><td>Unit time-related operating cost.</td></tr><tr><td>cf</td><td>Unit distance-related operating cost.</td></tr><tr><td>C(RT)</td><td>Terminal routing cost.</td></tr><tr><td>RT</td><td>Final route plan after the booking horizon ends.</td></tr><tr><td>E(RT)</td><td>Arc set of route plan RT.</td></tr><tr><td>N(RT)</td><td>Node set of route plan RT.</td></tr><tr><td>ωi,j</td><td>Travel time on arc (i,j).</td></tr><tr><td>di,j</td><td>Travel distance on arc (i,j).</td></tr><tr><td>li</td><td>Dwell time at node i.</td></tr><tr><td>Pr(·)</td><td>Probability operator.</td></tr><tr><td>λt</td><td>Passenger-arrival probability at epoch t.</td></tr><tr><td>μg</td><td>Probability that an arriving passenger belongs to segment g.</td></tr><tr><td>(ht,gt)</td><td>Observed information of the arriving passenger at epoch t.</td></tr><tr><td>st=Bt-1</td><td>System state at epoch t.</td></tr><tr><td>st∪{ (t,k)}</td><td>Updated state after passenger t chooses internal option k.</td></tr><tr><td>φ(st,k)</td><td>Spatiotemporal encoding of system state st.</td></tr><tr><td>Bt</td><td>Set of accepted bookings up to epoch t.</td></tr><tr><td>BT</td><td>Final accepted booking set at the booking deadline.</td></tr><tr><td rowspan="2">Choice-model notation</td><td>Walking distance for passenger t under option k.</td></tr><tr><td>Predicted in-vehicle travel time for passenger t under option k.</td></tr><tr><td>θtk</td><td>Predicted service position for passenger t under option k.</td></tr><tr><td>Utkg</td><td>Utility of internal option k for passenger t in segment g.</td></tr><tr><td>Utog</td><td>Utility of the outside option for passenger t in segment g.</td></tr><tr><td>εtk</td><td>Random disturbance term for internal option k.</td></tr><tr><td>εt0</td><td>Random disturbance term for the outside option.</td></tr><tr><td>βgwalk</td><td>Walking-distance sensitivity parameter for segment g.</td></tr><tr><td>βgprice</td><td>Price-sensitivity parameter for segment g.</td></tr><tr><td>βgivt</td><td>In-vehicle-travel-time sensitivity parameter for segment g.</td></tr><tr><td>V0</td><td>Baseline utility of the outside option.</td></tr><tr><td>Ptkg(at)</td><td>Probability that passenger t in segment g chooses internal option k.</td></tr><tr><td>Ptog(at)</td><td>Probability that passenger t in segment g chooses the outside option.</td></tr><tr><td rowspan="2">Dynamic programming and profit notation</td><td>Expected one-period reward at epoch t.</td></tr><tr><td>Value function at epoch t under state st.</td></tr><tr><td>VT(st)</td><td>Total net profit associated with the final accepted booking set BT.</td></tr><tr><td>Π(BT)</td><td>Opportunity cost of assigning the current passenger to option k.</td></tr><tr><td>ΔVk(st)</td><td>Downstream profit function under decision a and true cost vector c.</td></tr><tr><td>R(a;c)</td><td>Cost-independent revenue component in the reformulated downstream objective.</td></tr><tr><td>R0(a)</td><td>Cost-independent revenue component in the reformulated downstream objective.</td></tr><tr><td>Prediction and decision-focused learning notation</td><td>Dual-output prediction network.</td></tr></table>

$$
\boldsymbol {c} _ {a u g} = [ \boldsymbol {c} ^ {\top}, 1 ] ^ {\top}
$$

$$
\hat {\boldsymbol {c}} _ {a u g} = [ \hat {\boldsymbol {c}} ^ {\top}, 1 ] ^ {\top}
$$

$$
\boldsymbol {y} = \boldsymbol {c} _ {a u g}, \hat {\boldsymbol {y}} = \hat {\boldsymbol {c}} _ {a u g}
$$

$$
\boldsymbol {P} (\boldsymbol {a}) = \left[ P _ {k} (\boldsymbol {a}) \right] _ {k \in O}
$$

$$
\boldsymbol {w} (\boldsymbol {a}) = \left[ \boldsymbol {P} (\boldsymbol {a}) ^ {\top}, - R _ {0} (\boldsymbol {a}) \right] ^ {\top}
$$

$$
\mathcal {W}
$$

$$
z ^ {*} (\mathbf {y})
$$

$$
\boldsymbol {w} ^ {*} (\boldsymbol {y})
$$

$$
\ell_ {S P O}
$$

$$
\ell_ {S P O +}
$$

Augmented true cost vector.

Augmented predicted cost vector.

Compact notation used in the SPO derivation.

Choice-probability vector by decision ??.

Lifted decision vector induced by decision ??.

Convex hull of all lifted feasible decisions.

Optimal objective value under augmented cost vector ??.

Optimal lifted decision under augmented cost vector ??.

SPO regret loss.

SPO+ surrogate loss.

# Appendix B.

Starting from Eq. (6), consider a passenger arriving at epoch $t$ with option set $O _ { t }$ and price vector $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \mathbf \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \mathbf \mathbf \Psi \mathbf \Psi \mathbf \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \mathbf \Psi \mathbf \Psi \mathbf \mathbf \Psi \mathbf \mathbf \Psi \mathbf \Psi \mathbf \mathbf \Psi \mathbf \mathbf \Psi \mathbf \mathbf \Psi $ $( a _ { t k } ) _ { k \in O _ { t } } .$ .

If the passenger chooses internal option $k \in O _ { t }$ , the immediate reward is $f _ { e } + a _ { t k }$ . The next state becomes $s _ { t + 1 }$ , so the continuation value is $V _ { t + 1 } ( s _ { t + 1 } )$ . Therefore, the value of accepting the order through option $k$ is

$$
\left(f _ {e} + a _ {t k}\right) + V _ {t + 1} \left(s _ {t + 1}\right). \tag {B.1}
$$

If the order is not accepted, or if the outside option is chosen, the state remains $s _ { t }$ and the continuation value is $V _ { t + 1 } ( s _ { t } )$ . The net gain from accepting option $k$ is therefore

$$
[ f _ {e} + a _ {t k} + V _ {t + 1} (s _ {t + 1}) ] - V _ {t + 1} (s _ {t}) = (f _ {e} + a _ {t k}) - [ V _ {t + 1} (s _ {t}) - V _ {t + 1} (s _ {t + 1}) ]. \qquad (B. 2)
$$

We define the opportunity cost

$$
\Delta V _ {k} \left(s _ {t}\right) = V _ {t + 1} \left(s _ {t}\right) - V _ {t + 1} \left(s _ {t + 1}\right). \tag {B.3}
$$

This quantity measures the expected future profit loss caused by using capacity for the current order. Under the MNL model, the probability that a passenger from segment $g$ chooses option $k$ under price vector $\pmb { a } _ { t }$ is $P _ { t k g } ( \pmb { a } _ { t } )$ . Therefore, the expected net profit at epoch $t$ is

$$
\sum_ {k \in O _ {t}} P _ {t k g} \left(\boldsymbol {a} _ {t}\right) \left(f _ {e} + a _ {t k} - \Delta V _ {k} \left(s _ {t}\right)\right). \tag {B.4}
$$

Given $\boldsymbol { \varDelta V _ { k } } ( s _ { t } )$ , maximizing the original Bellman equation is equivalent to maximizing the expected net profit of the current passenger at each arrival. This yields the online pricing problem

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} P _ {t k g} \left(\boldsymbol {a} _ {t}\right) \left(f _ {e} + a _ {t k} - \Delta V _ {k} \left(s _ {t}\right)\right). \tag {B.5}
$$

The true opportunity cost $\Delta V _ { k } ( s _ { t } )$ is not observable in real time, because it depends on all future arrivals and on the final routing plan. We approximate it by the true marginal insertion cost $c _ { t k }$ , i.e., the change in total routing cost caused by removing option $k$ from the final route and re-optimizing. Formally,

$$
c _ {t k} = C \left(R ^ {T}\right) - C \left(R _ {- t k} ^ {T}\right). \tag {B.6}
$$

Substituting this approximation yields the per-passenger expected-profit maximization problem stated in Eq. (13) of the main text:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - c _ {t k}\right) P _ {t k g} \left(\boldsymbol {a} _ {t}\right). \tag {B.7}
$$

In real-time implementation, $c _ { t k }$ is replaced by its prediction $\hat { c } _ { t k }$ , which yields Eq. (14) in Section 4.2.3.

# Appendix C.

We depict the interior solution of the unconstrained pricing problem. For simplicity, we fix a specific passenger segment $g$ and an epoch $t$ , and omit these subscripts. Let $O$ denote the set of internal options, and 0 denote the outside option. For each $k \in O$ , define

$$
\eta_ {k} = - \beta_ {g} ^ {i v t} \tau_ {k} - \beta_ {g} ^ {w a l k} d _ {k}, \quad \beta = \beta_ {g} ^ {p r i c e} > 0. \tag {C.1}
$$

Then the MNL choice probabilities in Eq. (9) and Eq. (10) can be written as

$$
P _ {k} (\boldsymbol {a}) = \frac {\exp \left(\eta_ {k} - \beta a _ {k}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta a _ {j}\right)}, k \in O, \tag {C.2}
$$

and

$$
P _ {0} (\boldsymbol {a}) = \frac {\exp \left(V _ {0}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta a _ {j}\right)}. \tag {C.3}
$$

Given a true marginal insertion cost vector $\pmb { c } = ( c _ { k } ) _ { k \in O }$ , the pricing problem in Section 4.2.3 is

$$
\max  _ {a \in \mathbb {R} ^ {| O |}} R (\boldsymbol {a}; \boldsymbol {c}) = \max  _ {a \in \mathbb {R} ^ {| O |}} \sum_ {k \in O} \left(f _ {e} + a _ {k} - c _ {k}\right) P _ {k} (\boldsymbol {a}). \tag {C.4}
$$

Let

$$
g _ {k} = f _ {e} + a _ {k} - c _ {k} \tag {C.5}
$$

denote the option-specific net profit if option $k$ is selected. Under the MNL model, the derivatives of the choice probabilities with respect to the price of option $\ell \in O$ are

$$
\frac {\partial P _ {k} (\boldsymbol {a})}{\partial a _ {\ell}} = \left\{ \begin{array}{c c} - \beta P _ {k} (\boldsymbol {a}) \left(1 - P _ {k} (\boldsymbol {a})\right), & k = \ell , \\ \beta P _ {k} (\boldsymbol {a}) P _ {\ell} (\boldsymbol {a}), & k \neq \ell . \end{array} \right. \tag {C.6}
$$

Using Eq. (C.4), the derivative of the objective with respect to $a _ { \ell }$ is

$$
\frac {\partial R (\boldsymbol {a} ; \boldsymbol {c})}{\partial a _ {\ell}} = P _ {\ell} (\boldsymbol {a}) + \sum_ {k \in O} g _ {k} \frac {\partial P _ {k} (\boldsymbol {a})}{\partial a _ {\ell}} = P _ {\ell} (\boldsymbol {a}) [ 1 + \beta (R (\boldsymbol {a}; \boldsymbol {c}) - g _ {\ell}) ]. \tag {C.7}
$$

At any interior optimum ${ { \pmb a } } ^ { * } ( { \pmb c } )$ with $P _ { \ell } ( { \pmb a } ^ { * } ) > 0$ , the first-order condition $\partial R / \partial a _ { \ell } = 0$ implies

$$
g _ {\ell} = R \left(\boldsymbol {a} ^ {*}; \boldsymbol {c}\right) + \frac {1}{\beta}, \quad \forall \ell \in O. \tag {C.8}
$$

Eq. (C.8) shows that, at any interior optimum, the option-specific net profit is equalized across all internal options. Therefore, there exists a common markup term $m ( \pmb { c } )$ such that

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + m (\boldsymbol {c}), \quad \forall k \in O. \tag {C.9}
$$

Substituting Eq. (C.9) into Eq. (C.2), we obtain

$$
P _ {k} (m; \boldsymbol {c}) = \frac {\exp \left(\eta_ {k} - \beta c _ {k} + \beta f _ {e} - \beta m\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta c _ {j} + \beta f _ {e} - \beta m\right)}. \tag {C.10}
$$

Define

$$
B (\boldsymbol {c}) = \sum_ {j \in O} \exp \left(\eta_ {j} - \beta c _ {j}\right). \tag {C.11}
$$

Then the total internal-choice probability is

$$
S (m; \boldsymbol {c}) = \sum_ {k \in O} P _ {k} (m; \boldsymbol {c}) = \frac {\exp \left(\beta f _ {e} - \beta m\right) B (\boldsymbol {c})}{\exp \left(V _ {0}\right) + \exp \left(\beta f _ {e} - \beta m\right) B (\boldsymbol {c})}. \tag {C.12}
$$

Because Eq. (C.9) implies $f _ { e } + a _ { k } ^ { * } ( { \pmb c } ) - c _ { k } = m ( { \pmb c } )$ , the expected profit at any interior optimum can be rewritten as

$$
R \left(\boldsymbol {a} ^ {*} (\boldsymbol {c}); \boldsymbol {c}\right) = m (\boldsymbol {c}) S (m (\boldsymbol {c}); \boldsymbol {c}). \tag {C.13}
$$

Hence the multidimensional pricing problem reduces to

$$
\max  _ {m} m S (m; \boldsymbol {c}). \tag {C.14}
$$

Differentiating Eq. (C.12) with respect to ?? gives

$$
\frac {\partial S (m ; \boldsymbol {c})}{\partial m} = - \beta S (m; \boldsymbol {c}) \left(1 - S (m; \boldsymbol {c})\right). \tag {C.15}
$$

Therefore, the first-order condition for Eq. (C.14) is

$$
0 = \frac {d}{d m} [ m S (m; \boldsymbol {c}) ] = S (m; \boldsymbol {c}) - \beta m S (m; \boldsymbol {c}) (1 - S (m; \boldsymbol {c})). \tag {C.16}
$$

For any interior optimum with $S ( m ; c ) > 0$ , Eq. (C.16) at $m = m ( \pmb { c } )$ yields

$$
m (\boldsymbol {c}) = \frac {1}{\beta \left(1 - S \left(m (\boldsymbol {c}); \boldsymbol {c}\right)\right)} = \frac {1}{\beta P _ {0} \left(\boldsymbol {a} ^ {*} (\boldsymbol {c})\right)}. \tag {C.17}
$$

Equivalently, using Eq. (C.12), the common markup solves the scalar fixed-point equation

$$
m (\boldsymbol {c}) = \frac {1}{\beta} \left[ 1 + \exp \left(\beta f _ {e} - V _ {0} - \beta m (\boldsymbol {c})\right) B (\boldsymbol {c}) \right]. \tag {C.18}
$$

Eq. (C.18) can be converted into a standard form to solve using the Lambert W function. We rewrite

$$
\beta m (\boldsymbol {c}) - 1 = \exp \left(\beta f _ {e} - V _ {0} - \beta m (\boldsymbol {c})\right) B (\boldsymbol {c}). \tag {C.19}
$$

Multiplying both sides by $e x p ( \beta m ( \pmb { c } ) )$ yields

$$
(\beta m (\boldsymbol {c}) - 1) \exp (\beta m (\boldsymbol {c})) = \exp (\beta f _ {e} - V _ {0}) B (\boldsymbol {c}). \tag {C.20}
$$

Now define

$$
z = \beta m (\boldsymbol {c}) - 1. \tag {C.21}
$$

Then $\beta m ( \pmb { c } ) = \ b { z } + 1$ , and Eq. (C.20) becomes

$$
z \cdot e x p (z + 1) = e x p \left(\beta f _ {e} - V _ {0}\right) B (\boldsymbol {c}). \tag {C.22}
$$

Equivalently,

$$
z \cdot \exp (z) = \exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c}). \tag {C.23}
$$

By the definition of the Lambert W function, $W _ { 0 } ( x ) e x p ( W _ { 0 } ( x ) ) = x$ , we obtain

$$
z = W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right). \tag {C.24}
$$

Substituting back $z = \beta m ( \pmb { c } ) - 1$ yields

$$
m (\boldsymbol {c}) = \frac {1}{\beta} \left[ 1 + W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right) \right]. \tag {C.25}
$$

Hence, the unconstrained optimal price vector is

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + \frac {1}{\beta} \left[ 1 + W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right) \right], \quad \forall k \in O. \tag {C.26}
$$

Finally, combining Eq. (C.9) and Eq. (C.17), the optimal price vector has the form

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + m (\boldsymbol {c}), \quad m (\boldsymbol {c}) = \frac {1}{\beta P _ {0} \left(\boldsymbol {a} ^ {*} (\boldsymbol {c})\right)}, \quad \forall k \in O. \tag {C.27}
$$

In the online application, however, prices are limited to the feasible set $\mathcal { A } = \{ \pmb { a } _ { t } \colon - 1 0 \leq a _ { t k } \leq$ $1 0 , \forall k \in O _ { t } \}$ outlined in Section 4.2.3.

# References

Aflaki, A., & Zhang, Q. (2026). Is Your Price Personalized? Alleviating Customer Concerns with Inventory Availability Information. Operations Research, 74(1), 181-198.   
Akkerman, F., Dieter, P., & Mes, M. (2025). Learning dynamic selection and pricing of out-of-home deliveries. Transportation Science, 59(2), 250-278.   
Asdemir, K., Jacob, V. S., & Krishnan, R. (2009). Dynamic pricing of multiple home delivery options. European Journal of Operational Research, 196(1), 246-257.   
Bai, J., So, K. C., Tang, C. S., Chen, X., & Wang, H. (2019). Coordinating supply and demand on an ondemand service platform with impatient customers. Manufacturing & Service Operations Management, 21(3), 556-570.

Baier, M. J., Sörensen, L., & Schlüter, J. C. (2024). How successful is my DRT system? A review of different parameters to consider when developing flexible public transport systems. Transport Policy, 159, 130-142.   
Bardaka, E., Hajibabai, L., & Singh, M. P. (2020). Reimagining ride sharing: Efficient, equitable, sustainable public microtransit. IEEE Internet Computing, 24(5), 38-44.   
Bills, T. S., Twumasi-Boakye, R., Broaddus, A., & Fishelson, J. (2022). Towards transit equity in Detroit: An assessment of microtransit and its impact on employment accessibility. Transportation Research Part D: Transport and Environment, 109, 103341.   
Chandakas, E. (2020). On demand forecasting of demand-responsive paratransit services with prior reservations. Transportation Research Part C: Emerging Technologies, 120, 102817.   
Chen, Q., Wang, F., Hodge, B. M., Zhang, J., Li, Z., Shafie-Khah, M., & Catalão, J. P. (2017). Dynamic price vector formation model-based automatic demand response strategy for PV-assisted EV charging stations. IEEE Transactions on Smart Grid, 8(6), 2903-2915.   
Coutinho, F. M., van Oort, N., Christoforou, Z., Alonso-González, M. J., Cats, O., & Hoogendoorn, S. (2020). Impacts of replacing a fixed public transport line by a demand responsive transport system: Case study of a rural area in Amsterdam. Research in Transportation Economics, 83, 100910.   
Cui, S., Li, K., Yang, L., & Wang, J. (2022). Slugging: Casual carpooling for urban transit. Manufacturing & Service Operations Management, 24(5), 2516-2534.   
Currie, G., & Fournier, N. (2020). Why most DRT/Micro-Transits fail–What the survivors tell us about progress. Research in Transportation Economics, 83, 100895.   
Den Boer, A. V. (2015). Dynamic pricing and learning: Historical origins, current research, and new directions. Surveys in operations research and management science, 20(1), 1-18.   
D , , J , B , & T , L ‐ taxis. Production and Operations Management, 32(12), 3801-3815.   
Dong, L., Kouvelis, P., & Tian, Z. (2009). Dynamic pricing and inventory control of substitute products. Manufacturing & Service Operations Management, 11(2), 317-339.   
Du, C., He, F., & Lin, X. (2025). Dynamic pricing for air cargo revenue management. Transportation Research Part E: Logistics and Transportation Review, 197, 104088.   
, N , & G , “ , ” Management Science, 68(1), 9- 26.   
Fielbaum, A., Bai, X., & Alonso-Mora, J. (2021). On-demand ridesharing with optimized pick-up and drop-off walking locations. Transportation research part C: emerging technologies, 126, 103061.   
Galiullina, A., Mutlu, N., Kinable, J., & Van Woensel, T. (2024). Demand steering in a last-mile delivery problem with home and pickup point delivery options. Transportation Science, 58(2), 454-473.   
G ž č, N , G č , , & , O T commuting. European Transport Research Review, 17(1), 14.   
Ghimire, S., Bardaka, E., Monast, K., Wang, J., & Wright, W. (2024). Policy, management, and operation practices in US microtransit systems. Transport Policy, 145, 259-278.   
, N , ć, N , K j , R , ö , , & ä , J K drive us? Ex post evaluation of on-demand micro-transit pilot in the Helsinki capital region. Research in Transportation Business & Management, 32, 100390.   
Han, Z., Chen, Y., Li, H., Zhang, K., & Sun, J. (2019). Customized bus network design based on individual reservation demands. Sustainability, 11(19), 5535.   
Jang, M., Lee, S., Kim, J., & Kim, J. (2025). Dynamic minimum service level of demand–responsive transit: a prospect theory approach. Sustainability, 17(7), 3171.

Kaddoura, I., Leich, G., & Nagel, K. (2020). The impact of pricing and service area design on the modal shift towards demand responsive transit. Procedia Computer Science, 170, 807-812.   
Kayikci, Y., Demir, S., Mangla, S. K., Subramanian, N., & Koc, B. (2022). Data-driven optimal dynamic pricing strategy for reducing perishable food waste at retailers. Journal of cleaner production, 344, 131068.   
Kersting, M., Kallbach, F., & Schlüter, J. C. (2021). For the young and old alike–An analysis of the ’ -to-door DRT system EcoBus in rural Germany. Journal of Transport Geography, 96, 103173.   
Knierim, L., & Schlüter, J. C. (2021). The attitude of potentially less mobile people towards demand responsive transport in a rural area in central Germany. Journal of Transport Geography, 96, 103202.   
Kostic, B., Loft, M. P., Rodrigues, F., & Borysov, S. S. (2021). Deep survival modelling for shared mobility. Transportation Research Part C: Emerging Technologies, 128, 103213.   
Lee, E., Cen, X., Lo, H. K., & Ng, K. F. (2021). Designing zonal-based flexible bus services under stochastic demand. Transportation Science, 55(6), 1280-1299.   
Lei, D., Xu, M., & Wang, S. (2024). A conditional diffusion model for probabilistic estimation of traffic states at sensor-free locations. Transportation Research Part C: Emerging Technologies, 166, 104798.   
Lei, D., Xu, M., & Wang, S. (2025). A deep multimodal network for multi-task trajectory prediction. Information Fusion, 113, 102597.   
Li, Y., Li, X., & Zhang, S. (2021). Optimal pricing of customized bus services and ride-sharing based on a competitive game model. Omega, 103, 102413.   
Li, Y., Sun, H., Chang, X., Lv, Y., & Gao, K. Towards Smarter On-Demand Ride-Sharing: Balancing Efficiency and Flexibility Via Delayed Matching and Walking. Available at SSRN 5346078.   
Liu, T., & Ceder, A. A. (2015). Analysis of a new public-transport-service concept: Customized bus in China. Transport Policy, 39, 63-76.   
Liu, X., Chen, X., Potoglou, D., Tian, M., & Fu, Y. (2023). Travel impedance, the built environment, and customized-bus ridership: A stop-to-stop level analysis. Transportation Research Part D: Transport and Environment, 122, 103889.   
Lyu, Y., Chow, C. Y., Lee, V. C., Ng, J. K., Li, Y., & Zeng, J. (2019). CB-Planner: A bus line planning framework for customized bus systems. Transportation Research Part C: Emerging Technologies, 101, 233-253.   
Mayaud, J. (2025). On the role of microtransit in shaping new mobility patterns. Travel Behaviour and Society, 41, 101065.   
Miah, M. M., Naz, F., Hyun, K. K., Mattingly, S. P., Cronley, C., & Fields, N. (2020). Barriers and opportunities for paratransit users to adopt on-demand micro transit. Research in transportation economics, 84, 101001.   
Miller, P., de Barros, A. G., Kattan, L., & Wirasinghe, S. C. (2016). Analyzing the sustainability performance of public transit. Transportation Research Part D: Transport and Environment, 44, 177- 198.   
Naumov, S., & Keith, D. (2023). Optimizing the economic and environmental benefits of ride‐hailing and pooling. Production and Operations Management, 32(3), 904-929.   
Pellegrini, A., & Fielbaum, A. (2025). Are users ready to accept fully flexible walking in on-demand mobility?. Transportation Research Part C: Emerging Technologies, 178, 105210.   
Quadrifoglio, L., Dessouky, M. M., & Ordóñez, F. (2008). A simulation study of demand responsive transit system design. Transportation Research Part A: Policy and Practice, 42(4), 718-737.

Rath, S., Liu, B., Yoon, G., & Chow, J. Y. (2023). Microtransit deployment portfolio management using simulation-based scenario data upscaling. Transportation Research Part A: Policy and Practice, 169, 103584.   
Riggs, W., & Pande, A. (2022). On-demand microtransit and paratransit service using autonomous vehicles: Gaps and opportunities in accessibility policy. Transport Policy, 127, 171-178.   
Saharan, S., Bawa, S., & Kumar, N. (2020). Dynamic pricing techniques for Intelligent Transportation System in smart cities: A systematic review. Computer Communications, 150, 603-625.   
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.   
Shang, H., Chang, Y., Huang, H., & Zhao, F. (2022). Integration of conventional and customized bus services: An empirical study in Beijing. Physica A: Statistical Mechanics and its Applications, 605, 127971.   
Shu, W., & Li, Y. (2022). A novel demand-responsive customized bus based on improved ant colony optimization and clustering algorithms. IEEE Transactions on Intelligent Transportation Systems, 24(8), 8492-8506.   
Sörensen, L., Bossert, A., Jokinen, J. P., & Schlüter, J. (2021). How much flexibility does rural public transport need?–Implications from a fully flexible DRT system. Transport Policy, 100, 5-20.   
Sultana, Z., Mishra, S., Cherry, C. R., Golias, M. M., & Jeffers, S. T. (2018). Modeling frequency of rural demand response transit trips. Transportation Research Part A: Policy and Practice, 118, 494-505.   
Tong, L. C., Zhou, L., Liu, J., & Zhou, X. (2017). Customized bus service design for jointly optimizing passenger-to-vehicle assignment and vehicle routing. Transportation Research Part C: Emerging Technologies, 85, 451-475.   
Wang, J., Liu, K., Yamamoto, T., Wang, D., & Lu, G. (2023). Built environment as a precondition for demand-responsive transit (DRT) system survival: Evidence from an empirical study. Travel Behaviour and Society, 30, 271-280.   
Wang, L., Zeng, L., Ma, W., & Guo, Y. (2021). Integrating passenger incentives to optimize routing for demand-responsive customized bus systems. Ieee Access, 9, 21507-21521.   
Wu, B., Zuo, X., Chen, G., Ai, G., & Wan, X. (2024). Multi-agent deep reinforcement learning based realtime planning approach for responsive customized bus routes. Computers & Industrial Engineering, 188, 109840.   
Wu, W., Zhang, Z., Lu, K., & Ren, J. (2025). Dynamic demand-responsive transit scheduling with timedependent travel times: A joint supply and demand management approach. Transportation Research Part E: Logistics and Transportation Review, 202, 104232.   
Xia, D., Zheng, L., Cai, X., Liu, W., & Sun, D. (2022). Urban customized bus design for private car commuters. IEEE Internet of Things Journal, 9(21), 21723-21735.   
Yan, J., Martin, S., & Taylor, S. J. (2025). Trading flexibility for adoption: From dynamic to static walking in ride-sharing. Management Science, 71(7), 5875-5892.   
Yang, X., Strauss, A. K., Currie, C. S., & Eglese, R. (2016). Choice-based demand management and vehicle routing in e-fulfillment. Transportation science, 50(2), 473-488.   
Yu, C., Lin, H., Chen, Y., Yang, C., Yin, A., & Yuan, Q. (2024). Creating most needed customized bus services: A collaborative analysis of user-route dynamics. Transportation Research Part D: Transport and Environment, 133, 104312.   
Yu, C., Lin, H., Chen, Y., Yang, C., Yin, A., & Yuan, Q. (2024). Creating most needed customized bus services: A collaborative analysis of user-route dynamics. Transportation Research Part D: Transport and Environment, 133, 104312.

Zhang, W., Jacquillat, A., Wang, K., & Wang, S. (2023). Routing optimization with vehicle–customer coordination. Management Science, 69(11), 6876-6897.   
Zhang, Y., Luo, S., Zu, A., Ji, H., Kang, L., & Shao, C. (2025). Optimal design of fixed-route and demandresponsive transit with a dynamic stop strategy. Transportation Research Part A: Policy and Practice, 199, 104581.   
Zhao, Z., & Lee, C. K. (2021). Dynamic pricing for EV charging stations: A deep reinforcement learning approach. IEEE Transactions on Transportation Electrification, 8(2), 2456-2468.   
Zheng, Y., Li, W., Qiu, F., & Wei, H. (2019). The benefits of introducing meeting points into flex-route transit services. Transportation Research Part C: Emerging Technologies, 106, 98-112.