# Learning Dynamic Selection and Pricing of Out-of-Home Deliveries

Fabian Akkerman,a,* Peter Dieter,b Martijn Mesa

a University of Twente, 7522 NB Enschede, Netherlands; b Paderborn University, 33098 Paderborn, Germany

*Corresponding author

Contact: f.r.akkerman@utwente.nl, https://orcid.org/0000-0001-8055-9864 (FA); peter.dieter@upb.de,

https://orcid.org/0000-0003-0278-8068 (PD); m.r.k.mes@utwente.nl, https://orcid.org/0000-0001-9676-5259 (MM)

Received: November 23, 2023

Revised: July 10, 2024

Accepted: September 16, 2024

Published Online in Articles in Advance:

November 5, 2024

https://doi.org/10.1287/trsc.2023.0434

Copyright: $\circledcirc$ 2024 INFORMS

Abstract. Home delivery failures, traffic congestion, and relatively large handling times have a negative impact on the profitability of last-mile logistics. A potential solution is the delivery to parcel lockers or parcel shops, denoted by out-of-home (OOH) delivery. In the academic literature, models for OOH delivery are so far limited to static settings, contrasting with the sequential nature of the problem. We model the sequential decision-making problem of which OOH location to offer against what incentive for each incoming customer, taking into account future customer arrivals and choices. We propose dynamic selection and pricing of OOH (DSPO), an algorithmic pipeline that uses a novel spatialtemporal state encoding as input to a convolutional neural network. We demonstrate the performance of our method by benchmarking it against two state-of-the-art approaches. Our extensive numerical study, guided by real-world data, reveals that DSPO can save 19.9 percentage points $( \% \mathrm { { ( \% ) } } \lambda )$ in costs compared with a situation without OOH locations, 7%pt compared with a static selection and pricing policy, and $3 . 8 \% \mathrm { p t }$ compared with a state-of-the-art demand management benchmark. We provide comprehensive insights into the complex interplay between OOH delivery dynamics and customer behavior influenced by pricing strategies. The implications of our findings suggest that practitioners adopt dynamic selection and pricing policies.

History: This paper has been accepted for the Transportation Science special issue on TSL Conference 2023.

Funding: This work was supported by TKI DINALOG.

Keywords: out-of-home delivery • parcel lockers • demand management • machine learning • convolutional neural networks

# 1. Introduction

In 2021, the worldwide courier, express, and parcel (CEP) market was valued at $\$ 407.7$ billion with an estimated annual growth rate of $6 . 3 \%$ (Allied Market Research 2022). Last-mile delivery, particularly in urban areas, accounts for $4 0 \% { - } 5 0 \%$ of total CEP distribution costs. Factors such as traffic congestion, failed deliveries, and large handling times contribute significantly to these costs (Chen, Conway, and Cheng 2017, Dalla Chiara and Goodchild 2020, Dalla Chiara et al. 2021, Ranjbari et al. 2023). For instance, up to $1 0 \%$ of first delivery attempts fail, contributing to an average cost of $\$ 17$ per failure (Loquate 2021). In the United Kingdom alone, the estimated costs resulting from failed home deliveries exceed one billion dollars per year (Arnold et al. 2018). Moreover, failed deliveries have environmental consequences, notably increasing $\mathrm { C O } _ { 2 }$ emissions (Edwards et al. 2009). These delivery failures not only impose financial burdens on service providers, but also lead to customer dissatisfaction, negatively impacting the online shopping experience (Kedia, Kusumastuti,

and Nicholson 2017). Traffic congestion, unsuccessful deliveries, and handling time collectively contribute to approximately $2 8 \%$ of the costs and $2 5 \%$ of the emissions in the supply chain sector (Chen, Conway, and Cheng 2017).

To mitigate long driving and handling times and failed deliveries, an alternative is the implementation of out-ofhome (OOH) delivery systems. These systems involve delivering parcels to predetermined staffed locations or automated lockers from which customers can collect their items. The availability of OOH locations has increased by up to $3 6 \%$ annually in Europe (Last Mile Experts 2022), indicating a growing trend. OOH delivery can potentially save costs for logistics companies by reducing delivery failures and aggregating demand (Song et al. 2009, Savelsbergh and Van Woensel 2016). Previous studies show that OOH delivery can decrease driving distances by up to $6 0 \%$ (Enthoven et al. 2020) and is generally favored by customers (Ranjbari et al. 2023).

Integrating OOH delivery in last-mile logistics is an important and challenging research topic that provides

many future research directions on the strategic (e.g., OOH facility location), tactical (e.g., OOH location capacity, customer discounts), and operational (e.g., dynamic selection and pricing of OOH deliveries) decision-making levels. In this paper, we tackle the emergent challenge of dynamically offering OOH delivery options and providing dynamic prices (discounts or charges) to steer customer behavior. Aside from providing problem insights, we propose a dynamic selection and pricing policy with a novel spatial-temporal state encoding utilizing a convolutional neural network (CNN). We provide empirical evidence that our method outperforms existing state-of-the-art benchmarks across both small synthetic instances and a large real-world case from Seattle in the United States.

In the context of operational decision making for OOH deliveries, previous research focuses on static problem settings, in which all customer locations are known when decisions are made (see, e.g., Grabenschweiger et al. 2021, Mancini and Gansterer 2021, Zhang, Xu, and Wang 2023, Galiullina et al. 2024). However, in many real-world scenarios, customers arrive sequentially, and their delivery choices are subject to factors unknown to the route planner, for example, determined by personal preferences or past experiences.

In this paper, we attempt to fill this research gap by considering the dynamic decision of allocating OOH locations to newly arriving online customers and the possibility of influencing customer behavior by providing monetary incentives. The incentives can be positive or negative, representing a fee or a discount for the customer. The selection and pricing of OOH locations are by no means trivial decisions as we need to anticipate future customer arrivals, customer behavior, and the impact of discounts and charges on customer choice. Our problem is dynamic as we make selection and pricing decisions at each time step without having foresight of future customer arrivals. Consequently, we model our problem as a Markov decision process (MDP). Exact methods are not suitable for solving this MDP as the selection and pricing decision must be made swiftly for typical online retail platforms: a 100-millisecond delay in the load time of websites can decrease sales conversion by $7 \%$ (Akamai 2017). Currently, many retailers give customers complete freedom in selecting an OOH location, and in some countries, retailers offer a fixed customer discount for choosing an OOH delivery. In the future, if OOH delivery gets a larger share of total deliveries, smarter selection and pricing policies will be needed to fully exploit the potential of OOH delivery.

Our contribution to scientific literature is threefold. First, to our knowledge this paper is the first to formally define and study the OOH selection and pricing problem in a sequential decision-making context. Second, we compare several state-of-the-art solution methods from the literature and propose a novel machine

learning–based approach. Two benchmark policies are derived from time slot demand management literature (Yang et al. 2016) as this literature stream shares many similarities with our problem regarding dynamically arriving customer orders and intertwined pricing and routing decisions. However, our work focuses on the spatial dimension (where should customers be served?) rather than the temporal dimension (when should customers be served?) as studied before (see, e.g., Yang and Strauss 2017, Klein et al. 2018, Yildiz and Savelsbergh 2020). Furthermore, we benchmark against a proximal policy optimization (PPO) algorithm (Schulman et al. 2017), which is a state-of-the-art actor-critic reinforcement learning approach. As a third contribution, our extensive numerical study offers novel insights for both practitioners and researchers into both the problem and various solution methods. The numerical study is performed on synthetic problem instances (Gehring and Homberger 2002) as well as on instances derived from real-world data. We use publicly available Amazon order data of the greater Seattle area to construct these real-world instances (Merchan et al. 2022). Our code and used data can be found at https://github.com/ frakkerman/ooh_code.

The remainder of this paper is structured as follows. In Section 2, we present related work and outline the research gap. The problem and customer choice model are described in Section 3. The solution method is presented in Section 4. Section 5 includes the numerical study, and Section 6 concludes the paper.

# 2. Related Work

First, in Section 2.1, we introduce previous studies on last-mile logistics that consider demand management through offering or pricing of delivery options. Next, we discuss related work on OOH delivery in Section 2.2.

# 2.1. Demand Management in Last-Mile Logistics

Demand management is a key focus in last-mile logistics research, particularly regarding the offering and pricing of time slots for attended home delivery (AHD). The AHD demand management field is typically divided into the following categories: static time slot offering, dynamic time slot offering, differentiated time slot pricing, and dynamic time slot pricing (Agatz et al. 2011, Yang et al. 2016, Klein et al. 2019). Whereas restricting the time slot choice is seen as potentially leading to lost sales and customer dissatisfaction (Asdemir, Jacob, and Krishnan 2009), pricing is advocated as a more effective approach to balancing profits against lost sales.

In their study, Campbell and Savelsbergh (2006) explore the challenges of home delivery scheduling when customers are incentivized to choose different times of the day and broader delivery windows. Their

research is centered on developing algorithms to assess the viability and cost implications of various time slot options, taking into account existing customer requests. They operate under the assumption that the likelihood of customers selecting specific time slots is known in advance. Asdemir, Jacob, and Krishnan (2009) addresses this issue by considering a more advanced customer choice model, namely, a multinomial logit (MNL) model. Concerning routing costs, more recent research approximates these not solely based on requests already in the system, but also by anticipating future requests (Yang et al. 2016, Klein et al. 2018).

Currently, approximate dynamic programming approaches have been the method of choice in AHD pricing research; see, for instance, Yang and Strauss (2017), Koch and Klein (2020), and Vinsensius et al. (2020). Ulmer (2020) considers a same-day delivery problem in which booking and service horizons elapse simultaneously. By implementing an anticipatory pricing and routing policy, they incentivize customers to choose delivery deadlines that optimize fleet utilization, thereby increasing revenue and the number of sameday orders fulfilled. Strauss, Gu¨ lpinar, and Zheng (2021) explore the concept of flexible time windows, in which customers receive a financial incentive to be notified of their service window only shortly before dispatch. In contrast, Yildiz and Savelsbergh (2020) shift the focus to incentives for selecting delivery days rather than time windows within a day. Recent work also studies demand management in logistics outside the AHD context, for instance, the matching of requests in peerto-peer logistics, considering the finite resources of the offering party (Ausseil, Pazour, and Ulmer 2022, Karabulut, Gholizadeh, and Akhavan-Tabatabaei 2022). Collectively, these studies underscore a trend toward more dynamic and customer-focused solutions in demand management for last-mile logistics.

# 2.2. Last-Mile Logistics with OOH Delivery

OOH delivery is gaining attention in the scientific literature as it is already used in practice. On a strategic level, OOH research typically focuses on deciding on the location of OOH facilities and the capacity of such facilities. Facility location problems are often solved using exact or matheuristic approaches as demonstrated in Deutsch and Golany (2018); Xu et al. (2021); Kahr (2022); Luo, Ji, and Ji (2022); Lin et al. (2022); Lyu and Teo (2022); Mancini, Gansterer, and Triki (2023); and Raviv (2023). In summary, the recent OOH facility location literature concludes that (i) adopting OOH enables logistics service providers to cope with large demand increases; (ii) OOH locations should be situated in densely populated areas that are frequently traversed by vehicle routes; (iii) small and medium-sized OOH lockers are often preferred over larger, more costly lockers; and (iv) opening

too many OOH locations in an area can negatively affect profitability.

On a tactical and operational level, the literature considers heuristics for variants of the vehicle routing problem (VRP) that incorporate OOH locations, such as the work of Jiang et al. (2020); Pan et al. (2021); Dumez, Lehue´de´, and Pe´ton (2021); and Peng et al. (2023). Zhou et al. (2018) discuss a multidepot, two-echelon VRP in which customers have the option of home delivery or OOH delivery. Mancini and Gansterer (2021) and Grabenschweiger et al. (2021) explore the impacts of fixed discounts and heterogeneous locker boxes on customer acceptance in VRPs with OOH locations. Additionally, Ulmer and Streng (2019) address same-day delivery scenarios in which all customer requests require delivery to pickup stations. Emerging topics in operational OOH research include the use of alternative delivery modalities and mobile parcel lockers to increase customer service as discussed in studies by Enthoven et al. (2020), Ghaderi et al. (2022), Vukic´evic ´ et al. (2023), Schwerdfeger and Boysen (2022), and Liu et al. (2023). In most works, customer choice is modeled as a constant function in relation to the distance traveled to OOH locations (Grabenschweiger et al. 2021, Mancini and Gansterer 2021, Schwerdfeger and Boysen 2022, Janinhoff, Klein, and Scholz 2023, Peng et al. 2023) or using a rankingbased approach (Dumez, Lehue´de´, and Pe´ton 2021). Many studies do not model customer choice beyond a preference list or assume full control by the retailer (Arnold et al. 2018, Zhou et al. 2018, Sitek and Wikarek 2019, Enthoven et al. 2020). A few recent studies incorporate customer choice models for OOH delivery. Janinhoff and Klein (2023) model customer choice based on a random utility derived from the distance to delivery points, whereas Galiullina et al. (2024) model it using a price-based probability. Zhang, Xu, and Wang (2023) propose a convex optimization model to find the customer distribution across delivery options. Only the latter two studies consider pricing. For a comprehensive classification of recent OOH literature, we refer to Janinhoff et al. (2024).

Concluding, only a limited number of papers consider the selection and pricing of OOH to influence customer behavior. Furthermore, all described studies assume static settings and do not model a sequential decision process as likely experienced by many service providers, leading to a call for research that incorporates these realworld complexities. To the best of our knowledge, this is the first paper that considers the offering and pricing of OOH delivery in a sequential decision-making context, modeling the stochastic nature of customer behavior and customer arrivals, as observed in practice.

# 3. Model

In this section, we provide the model for the OOH selection and pricing problem. We provide the general

problem description in Section 3.1 and detail the customer choice model in Section 3.2. We end the section with an overview of all notation.

# 3.1. Problem Description

Our model of the OOH selection and pricing problem is motivated by the situation at online retailers when customers choose between home delivery or delivery at a parcel locker or shop. We consider a retailer that serves customers using a fixed fleet of vehicles. Before the delivery day, customers order products on the retailer’s website and, upon checkout, select a delivery option: home or OOH. We consider a fixed booking horizon $[ 0 , T ] ,$ where $T$ is the cutoff time, that is, the time after which no new customer can request delivery for the next day. During the booking horizon, customers $c _ { t } \in$ ${ \boldsymbol { B } } _ { T }$ arrive at discrete time steps $t ,$ and the number of customers that arrive on a given day follows an unknown distribution $D$ . We consider a situation in which different delivery options $k \in \mathcal { K }$ represent different costs, and the customer can be nudged to specific options by offering (i) only a subset of all delivery options in $\mathcal { O } \subseteq \kappa$ and (ii) discounts and charges per delivery option $k$ . The set of delivery options $\mathcal { \kappa }$ includes a home delivery option $h$ and a fixed set of OOH locations $\mathcal { L }$ . Thus, ${ \mathcal { K } } = \{ { \bar { h } } \} \cup { \mathcal { L } }$ . Without loss of generality and in line with retailer practice, we always offer home delivery as a delivery option. For example, if $\mathcal { L } = \{ \mathrm { O O H } 1 , \mathrm { O O H } 2 , \mathrm { O O H } 3 \} _ { }$ , then $\mathcal { K } = \{ h , \mathrm { O O H } 1 , \mathrm { O O H } 2 , \mathrm { O O H } 3 \}$ . The subset of offered options $\mathcal { O }$ might be $\{ h , \mathrm { O O H 1 } \} ,$ , meaning that the customer is offered the choice between home delivery and OOH location 1. We conduct experiments with finite and infinite capacity lockers. The remaining capacity of an OOH location l at time $t$ is denoted by $k _ { t , l }$ . Customers are divided into a finite set of segments g ∈ G with µg being $g \in { \mathcal { G } }$ $\mu _ { g }$ the probability that a customer from segment g arrives on an arbitrary time t. Customer choice behavior is modeled using an MNL model detailed in Section 3.2.

The main focus of this study is the pricing of delivery options, that is, the delivery charge or discount given to home and OOH delivery. However, we also consider

the selection of a subset of delivery options $\mathcal { O } \subseteq \kappa$ to be offered to the customer. We illustrate these decisions in Figure 1. Here, at a given time step $t ,$ five customers already arrived and accepted a delivery option. Three customers requested home delivery, and two customers chose to have their parcels delivered at an OOH location. Each OOH location has a finite capacity of five parcels; the number shown above the OOH location indicates the remaining capacity. Next, a new customer arrives, and this requires the retailer to offer different delivery options. The offered delivery locations get a delivery charge or discount. Here, the home delivery costs the customer a delivery charge of 2.5, whereas choosing one of the four offered OOH locations provides the customer a discount of 1.5 or 3.0, respectively. One OOH location is not offered to the customer.

An important characteristic of our problem is its dynamic nature. The retailer does not know how many customers will arrive before the cutoff time T, nor does it know the home locations of future customers. A customer that might seem remote and, therefore, expensive might receive a high delivery charge for home delivery and a high incentive for choosing an OOH delivery. However, as the booking horizon unfolds, more customers might appear close to this customer’s home location. This means that the customer, in hindsight, is not remote at all, and too high incentives were given.

We consider the following sources of costs: (i) costs related to operational time expressed in driver salaries $C ^ { w }$ , (ii) costs related to driving distance expressed in fuel costs ${ \dot { C } } ,$ (iii) costs related to providing discounts to customers $j _ { t } ^ { - }$ , and (iv) costs related to delivery failures $C ^ { m }$ . We assume drivers to be service providers who are paid per hour. We do not include fleet costs because we consider a fixed fleet of $V$ vehicles. Sources of revenue are the sales revenue per customer $r _ { t }$ and the collected delivery charges $j _ { t } ^ { + }$ . Note that the total costs and revenues are calculated after the booking horizon has ended at time T when all customers $B _ { T }$ have been revealed. In line with practice, we model the possibility of delivery failure. With probability $\mathbb { P } ^ { m }$ , home delivery will fail. Failed

![](images/b65db0303e2d6ebd9dcf547f26abdca9db77c16ba9b4eea1198fc810550d3653.jpg)  
Figure 1. Exemplary Situation During the Booking Horizon

![](images/1bb20bf2034635a176e11162899362841522e810c420ae9c87f9f0322db5b495.jpg)

● Booked home delivery   
$\bigcirc$ Customer who selected OOH delivery   
$\oslash$ New customer   
$\spadesuit$ Offered OOH location   
$\diamondsuit$ Non-offered OOH location

Delivery price

delivery results in a fixed monetary penalty of $C ^ { m }$ . We assume that delivery to an OOH location never fails. We do not consider delivery time windows. The goal of the retailer is to maximize the total profits.   
The sequence of events at a given time step $t$ is as follows:

1. A customer $c _ { t }$ arrives at time $t$ and fills the online basket.   
2. At the checkout, the customer gets to see different prices per delivery option $k \in \mathcal { O }$ .   
3. The customer selects a delivery option $k$ $( c _ { k , t } ^ { c h o i c e } =$ 1) and leaves the online system.   
4. The selected delivery location is added to $\boldsymbol { B } _ { t - 1 }$   
5. If applicable, the remaining capacity of the chosen OOH location is reduced.

During this event sequence, the decision maker has to decide on the selection and pricing at step 2. Our assumed sequence of events omits the possibility that a customer first chooses a delivery option before completing the basket. After the booking period, at cutoff time $T ,$ , the retailer plans routes serving all booked customers, after which the routes are executed by the fleet. The decision problem can be formulated as an MDP. We define a state space $S _ { \nu }$ , a decision space $\mathcal { A }$ , a cost function $c : S \times \mathcal { A }  \mathbb { R } ,$ , and transition dynamics $\mathbb { P } : \mathcal { S } \times \mathcal { A } \times \mathcal { S } $ [0, 1]. Here, the function $\mathbb { P }$ maps the probability of reaching a state to each state–decision pair. A state $s _ { t } \in S$ at time $t$ is represented by a discrete tuple. A state consists of

$$
s _ {t} = \left[ c _ {t}, \mathcal {B} _ {t - 1}, \vec {\mathbf {k}} _ {t} \right], \tag {1}
$$

where $c _ { t }$ represents the newly arrived customer, $\boldsymbol { B } _ { t - 1 }$ represents all booked delivery locations, and $\vec { \pmb { \kappa } } _ { t } = \{ k _ { t , l } |$ $l \in \mathcal { L } \}$ represents the vector of remaining capacities of all OOH locations. We use $a _ { k } ^ { s e l e c t i o n }$ to denote the binary decision of selecting a delivery location $k$ to be offered to the customer. Next, a pricing denotes the pricing decision for $a _ { k } ^ { p r i c i n g }$ this location. Each delivery price is bound by a maximum discount and maximum charge, $a _ { k } ^ { p r i c i n g } \in [ a , b ]$ a pricing . Here, a negative price indicates a discount, and a positive price indicates a delivery charge. We consider multidimensional decisions, represented as a vector of tuples and, for conciseness, referred to as a decision:

$$
\vec {a} _ {t} = \left[ a _ {k} ^ {\text {s e l e c t i o n}}, a _ {k} ^ {\text {p r i c i n g}} \right], \quad \forall k \in \mathcal {K}. \tag {2}
$$

The transition from state $s _ { t }$ to state $s _ { t + 1 }$ is done by using the information resulting from the decision and exogenous events. More precisely, we add the customer choice for a delivery location to the system, add the new stop, reduce remaining OOH capacity if applicable, and observe a new customer arrival. The cost or revenue from a discount or delivery charge at a given time step t can be calculated using

$$
j _ {t} = \sum_ {k \in \mathcal {K}} a _ {k, t} ^ {\text {p r i c i n g}} c _ {k, t} ^ {\text {c h o i c e}}, \tag {3}
$$

where $j _ { t } ^ { - } = - \mathrm { m i n } \{ j _ { t } , 0 \}$ denotes the cost from providing discounts and $j _ { t } ^ { + } = \operatorname* { m a x } \{ 0 , j _ { t } \}$ denotes revenue from delivery charges. The binary variable $c _ { k , t } ^ { c h o i c e }$ denotes the choice of the customer, where $k = h$ indicates home delivery.

The routing plan made after cutoff time $T$ is denoted by $R _ { T }$ . The routing dimension of our problem consists of a capacitated vehicle routing problem (CVRP). The directed graph $\mathcal { G } = ( \nu , \mathcal { E } )$ models the system of vertices $B _ { T } \cup \mathcal { W }$ with delivery and depot locations. The travel distance and time on an edge $( i , j ) \in \mathcal { E }$ is expressed by $d _ { i , j }$ and $w _ { i , j } ,$ respectively. Note that we model travel distance and time separately to account for variable travel speed and congestion. It is allowed to combine loads to the same location (one OOH location), but it is also allowed to split them over multiple vehicles. A vehicle always starts and ends at the same depot and has a fixed capacity of $K$ customers. A vehicle route must adhere to individual vehicle capacity constraints. A vehicle can only leave from a location i after the service duration $l _ { i } ,$ which accounts for the time needed for parking, walking, and delivering parcels. The service duration at the depot, $l _ { 0 } ,$ is zero.

We calculate the total costs, together with routing costs, after the cutoff time T. Hence, the objective is to maximize total profits on any given day:

$$
\begin{array}{l} \max \left(\sum_{t = 0}^{T}r_{t} + j_{t}^{+} - j_{t}^{-}\right) - C^{w}\left(\sum_{\substack{i\in R_{T}\\ j\neq i}}\sum_{\substack{j\in R_{T}\\ j\neq i}}w_{i,j} + l_{i}\right) \\ - C ^ {f} \left(\sum_ {i \in R _ {T}} \sum_ {\substack {j \in R _ {T} \\ j \neq i}} d _ {i, j}\right) - C ^ {m} \left[ \mathbb {P} ^ {m} \sum_ {t = 0} ^ {T} c _ {k = h, t} ^ {\text {choice}} \right]. \tag{4} \\ \end{array}
$$

We consider a policy $\pi : { \mathcal { S } }  A$ that defines the selection and pricing behavior. Our primary objective is to minimize costs as a key component of our analysis.

# 3.2. Customer Choice Model

Inspired by Yang et al. (2016), we model customer choice, $c _ { k , t } ^ { c h o i c e }$ , using the MNL model. For the MNL model, we assume customers are utility maximizers; that is, a customer always selects a delivery option that has the highest utility. Note that the pricing bounds $[ a , b ]$ ensure that prices remain realistic and reduce unfairness given the objective to minimize costs. The utility of a delivery option $k$ to customer segment $g$ is given by

$$
u _ {k, g} = - \beta_ {g} ^ {k} \exp \left[ d _ {0, k} \right] + \beta_ {g} ^ {d} a _ {k} ^ {\text {p r i c i n g}} + \epsilon . \tag {5}
$$

Similar to the prevailing focus in related literature, a delivery location’s utility is based on its distance from the customer’s home address. Hence, $\beta _ { g } ^ { k }$ is the sensitivity of the utility to the distance between the home address and the offered delivery location. The parameter $\boldsymbol { \beta } _ { g } ^ { d }$ represents the sensitivity of the utility to the delivery

price, denoted as $a _ { k } ^ { p r i c i n g }$ a pricingk . For simplicity, the subscript t is omitted. We ensure that ${ { u } _ { \{ 0 , k \} } }$ does not become too large, preventing the exponential term from dominating the utility. Similar to Lyu and Teo (2022), who study the OOH facility location problem, we split the utility of home delivery $( u _ { k , g }$ with $k = h$ ) from the utility given to OOH locations and consider it a tunable parameter. Apart from the deterministic components, we include $\epsilon ,$ , which is an independent and identically distributed (i.i.d.) random variable that follows the standard Gumbel distribution with $\mu = 0$ and $\beta = 1$ as is common for MNL models. Because we do not consider customer walk-aways, the probability that a customer chooses delivery option $k$ from all offered options $\mathcal { O }$ given the vector of delivery prices $\{ a _ { 0 } ^ { p r i c i n g } , a _ { 1 } ^ { p r i c i \hat { n } g } , . . . \} = \stackrel { \smile } { a } ^ { p r i c i n g }$ pricing is

$$
\mathbb {P} _ {k, g} (\vec {\boldsymbol {a}} ^ {p r i c i n g}) = \frac {\exp \left[ - \beta_ {g} ^ {k} \exp \left[ d _ {0 , k} \right] + \beta_ {g} ^ {d} a _ {k} ^ {p r i c i n g} \right]}{\sum_ {k \in \mathcal {O}} \exp \left[ - \beta_ {g} ^ {k} \exp \left[ d _ {0 , k} \right] + \beta_ {g} ^ {d} a _ {k} ^ {p r i c i n g} \right]}. \tag {6}
$$

To ease notation, we omit a base utility as it is absorbed by the sensitivity parameters. A base utility can be

added without loss of generality. For more details on the MNL choice model, we refer to Train (2009) and Yang et al. (2016). We end this section with a summary of all relevant notation; see Table 1.

# 4. Solution Method

Figure 2 depicts the used decision-making pipeline for dynamic selection and pricing of OOH delivery (DSPO). The pipeline entails two subdecisions forming the overall policy $\pi$ together. First, the state, containing the already booked customers, the newly arrived customer, and remaining OOH capacities, is used as input to obtain the heuristic selection decision $a ^ { s e l e c t i o n }$ . Next, the decision $a ^ { s e l e c t i o n }$ and state $s _ { t }$ are used to derive the pricing decision, which is obtained from a supervised machine learning model. Both decisions are combined in a joint selection and pricing policy, $\pi : { \mathcal { S } }  A$ . The remainder of this section details each step in the pipeline. We discuss selection and, subsequently, pricing in Section 4.1. Furthermore, we detail the training

Table 1. Summary of Notation   

<table><tr><td>Variable</td><td>Description</td></tr><tr><td colspan="2">General variables</td></tr><tr><td>T</td><td>The cutoff time after which no new customers can be booked</td></tr><tr><td>l∈L</td><td>The set of all OOH locations</td></tr><tr><td>h</td><td>The home delivery option offered to the customer</td></tr><tr><td>k∈K</td><td>The set of all delivery locations, K = L ∪ {h}</td></tr><tr><td>D</td><td>The CDF of the number of arriving customers</td></tr><tr><td>g∈G</td><td>The customer segments</td></tr><tr><td>μg</td><td>The probability that a customer arrives from segment g</td></tr><tr><td>V</td><td>The fleet size</td></tr><tr><td>K</td><td>The carrying capacity per vehicle per day</td></tr><tr><td>Cw</td><td>The salary costs per hour per driver</td></tr><tr><td>Cf</td><td>The fuel costs per distance unit</td></tr><tr><td>Cm</td><td>The fixed costs paid per delivery failure</td></tr><tr><td>j_t</td><td>The costs from a discount at time t</td></tr><tr><td>j_t+</td><td>The revenue from a delivery charge at time t</td></tr><tr><td>rt</td><td>The revenue from a customer sale at time t</td></tr><tr><td>RT</td><td>The final routing plan serving all customers</td></tr><tr><td>di,j</td><td>The travel distance on an edge connecting vertices i and j</td></tr><tr><td>wi,j</td><td>The travel time on an edge connecting vertices i and j</td></tr><tr><td>li</td><td>The service duration at location i</td></tr><tr><td>Pm</td><td>The probability of delivery failure at any home address</td></tr><tr><td>uk,g</td><td>The utility attributed to delivery option k for segment g</td></tr><tr><td>βk^g</td><td>The sensitivity of the utility for segment g to the distance between the home and the OOH location</td></tr><tr><td>βd^g</td><td>The sensitivity of the utility of segment g to the given price</td></tr><tr><td>ε</td><td>The i.i.d. random MNL noise per delivery option k</td></tr><tr><td>Pk,g(ˆpricing)</td><td>The probability a customer from segment g selects option k given the price vectorˆpricing</td></tr><tr><td colspan="2">State variables</td></tr><tr><td>ct</td><td>The customer arriving at time t</td></tr><tr><td>Bt-1</td><td>The booked delivery stops at time t-1</td></tr><tr><td>ˆk</td><td>The vector of remaining capacities of all OOH locations,ˆk = {kt,l | l ∈ L} at time t</td></tr><tr><td colspan="2">Policy variables</td></tr><tr><td>O←a selection(st)</td><td>The subdecision of selecting a set of delivery locations to offer given state st</td></tr><tr><td>a selection(st)←π selection(st)</td><td>The selection subpolicy given state st</td></tr><tr><td>a pricing(st, a selection(st))</td><td>The subdecision of pricing delivery locations given state st and the selection decision</td></tr><tr><td>a pricing(st)←π pricing(st, a selection(st))</td><td>The pricing subpolicy given state st</td></tr><tr><td>[a,b]</td><td>The bounds of delivery prices</td></tr><tr><td>a(st)←π</td><td>The joint policy of selection and pricing given state st</td></tr></table>

![](images/9c1a09e834226d420ca9ebe7e1544be623f801caf70aa7fecd8ad8d4c0dce5e8.jpg)  
Figure 2. Pipeline for Dynamic Selection and Pricing of Out-of-Home Delivery

procedure of the supervised machine learning model in Section 4.2 and discuss our algorithmic design choices in Section 4.3.

# 4.1. Selection and Pricing Decision

For the selection decision, we use a simple heuristic rule that selects a subset of OOH locations. To be precise, we select the N OOH locations closest to the customer’s home address that still have remaining capacity, that is, $k _ { t , l } > 0$ . We ensure that $N$ is sufficiently large to prevent a decline in the probability of customers choosing an OOH delivery option because of a limited offering as the utility of an OOH location is mainly determined by the proximity of an OOH location to the home address; see Section 3.2. We limit the offering to $N$ to reduce the number of computations, which is especially relevant when the total number of OOH locations is large. In Appendix C, we conduct a sensitivity analysis for different levels of N.

For the pricing decision, we employ a cost approximation that estimates the costs of adding a delivery location, whether a specific OOH location or the customer’s home address, to the delivery route. Note that we aim to estimate the impact of a stop on the final delivery schedule because a simple cheapest insertion in the current route with the so-far known stops does not suffice. In the remainder of this section, we, subsequently, detail (i) how we encode the state, (ii) the machine learning model used for delivery cost approximation, (iii) how we obtain training data, and (iv) how we obtain prices from the cost approximation.

4.1.1. State Encoding. Recall that the state consists of a set containing all currently known delivery locations, both OOH locations and home addresses. As is common in practice, we do not want to discriminate customers by providing different pricing schemes to customers with the same location. Therefore, we aggregate customer locations by counting the number of delivery locations in a specific aggregation area. We denote the state encoding by $\phi ( s _ { t } )$ . We divide the total service area into $M$ spatial areas. Note that we do not know in advance how many customers will arrive during the booking horizon. However, the number of served customers on a day has a major influence on the percustomer delivery costs. Therefore, apart from the spatial aggregation of the state, we also consider a temporal aggregation. We aggregate customers based on $D ^ { T }$

predefined and equal-length time intervals of the booking horizon. An exemplary spatial-temporal state encoding $\phi ( s _ { t } )$ is shown in Figure 3. Here, delivery locations are indicated by black dots, and the time intervals in which the customers arrived in the system $( D ^ { T } = \{ 0 , 1 , 2 \}$ ) are indicated with the numbers in the dots. Although we do not know up front how many customers will arrive on a specific day, this state encoding allows the CNN to estimate the arrival distribution. We consider both M and $D ^ { T }$ to be tunable hyperparameters. The results of hyperparameter tuning can be found in Appendix F. We provide the remaining capacity $k _ { t , l }$ of an OOH location as a separate variable to our prediction model. In an infinite capacity setting, this variable is omitted.

4.1.2. Cost Prediction Model. Our state encoding $\phi ( s _ { t } )$ is particularly suitable for CNNs. Unlike fully connected (FC) neural networks, CNNs take a multidimensional matrix as input and can abstract meaning from spatial relationships. In CNNs, convolution layers use kernel convolution, which is a process in which a small matrix, called a kernel or filter, transforms the data by passing over multiple input dimensions. Kernel convolution transforms data into activation maps, which are feature abstractions from the raw data. In CNNs, convolution layers are commonly followed by pooling layers, which allows the high-dimensional output data from the convolution operations to be reduced to a manageable dimension. Most CNNs use several convolution and pooling layers; see Li et al. (2022) for an overview of popular CNN architectures. An efficient way to learn nonlinear patterns from the CNNs output is to add FC layers after the final pooling layer. For a detailed explanation of CNNs, we refer to Goodfellow, Bengio, and Courville (2016). Although many architectures for CNNs exist, we use what we consider a vanilla architecture, consisting of two convolution layers, an average pooling layer, and two FC layers. Our CNN architecture is depicted in Figure 4. Remaining capacity is directly fed to the FC layers after the convolutional layers. We denote our neural network by $\mathcal { N } _ { \theta }$ , where $\theta$ are trainable weights. The neural network $\mathcal { N } _ { \theta }$ takes as input the encoded state $\phi ( s _ { t } )$ and outputs a single value in $\mathbb { R }$ . The goal of the neural network is to accurately predict the true ut unknown costs $C _ { k , t } ^ { t r u e }$ of inserting a elivery loca-$k$ $t$ route $R _ { T }$ given $\boldsymbol { B } _ { t - 1 }$ . Therefore, to obtain the expected

![](images/561ddd3bba0005ac7449e9ea08858b50f3b2561cfbb8d9408dbe3a4b1217e17d.jpg)  
Figure 3. The State Encoding for DSPO

costs of inserting a location $k ,$ , we calculate

$$
\hat {C} _ {k, t} ^ {D S P O} = \mathcal {N} _ {\theta} \left(\phi \left(s _ {t}, k\right)\right), \tag {7}
$$

where $\left( { { s _ { t } } , k } \right)$ is the current state with the potential new delivery location $k$ added. Note that we do inference for all offered delivery locations; that is, the DSPO architecture yields a cost prediction vector for each offered OOH location and the home location. The hyperparameter tuning results of the CNN are summarized in Appendix F.

4.1.3. Obtaining Training Data. For obtaining training data, we need to store features, that is, the encoded state $\phi ( s _ { t } )$ as depicted in Figure 3, and the actual costs related to this state, $C _ { k , t } ^ { t r u e }$ . For a given customer, these costs are obtained by removing the customer’s chosen delivery location from the final route $R _ { T } ,$ , obtaining a new route using a CVRP solver, and subsequently storing the difference in total routing time compared with the route that serves all customers. We calculate the costs related to $\phi ( s _ { t } )$ after the cutoff time $T$ when the ccurate $C _ { k , t } ^ { t r u e }$ $l _ { i }$

![](images/e1e78e3a653db22195225fdca55fbbf57efd9a4dcbc1eaa204ee2a22c4a5ff2f.jpg)  
Figure 4. (Color online) The CNN Architecture for DSPO

![](images/3b17852dcf8746780615f8482861e2645ea02c4f46b5412c488586e4e634d2d4.jpg)  
Figure 5. An Illustration of How Training Data Are Obtained During a Booking Horizon in Which Three Customers Arrive

customer, we can obtain the total costs of salary and fuel related to a delivery location i. When i is an OOH location, the costs related to $l _ { i }$ are divided over the number of customers utilizing this delivery option. We disregard possible costs because of delivery failure in $C _ { k , t } ^ { t r u e }$ as, in our model, delivery failures cannot be predicted based on spatial-temporal data. Note that the features are stored during the simulated booking horizon, but the actual costs related to the feature values are only obtained after the cutoff time. The process of obtaining data during a simulation is illustrated in Figure 5. For illustrative purpose, we assume that one customer arrives per time step t. Here, we solve four CVRPs: one with three customers and three CVRPs in which the first, second, and third customer is disregarded, respectively. This results in three insertion costs, which are the costs related to the stored features $\phi ( )$ .

4.1.4. From Cost Prediction to Delivery Prices. We use the insertion cost for all offered delivery locations, $C _ { k , t } ^ { t r u e } ,$ , to obtain optimal prices given the revenue, delivery costs, pricing revenue and costs, and expected sensitivity of customers to delivery prices. For this, we use the proof in Dong, Kouvelis, and Tian (2009). First, we simplify Equation (4) and solve the online decision problem, for simplicity, denoted without time t and customer segment g indices:

$$
\max  \sum_ {k \in \mathcal {O}} \left(r + a _ {k} ^ {p r i c i n g} - C _ {k} ^ {t r u e}\right) \mathbb {P} _ {k} \left(\vec {\boldsymbol {a}} ^ {p r i c i n g}\right). \tag {8}
$$

Equation (8) maximizes the profits and is concave in the customer selection probabilities as shown by theorem 1 in Dong, Kouvelis, and Tian (2009). We apply their result to obtain the following concave optimization problem:

$$
\max  \sum_ {k \in \mathcal {O}} \left(- \frac {1}{\beta^ {d}} \left(- \beta^ {k} \exp \left[ d _ {0, k} \right] - \ln \mathbb {P} _ {k}\right) + r - C _ {k} ^ {\text {t r u e}}\right) \mathbb {P} _ {k}. \tag {9}
$$

Note that $\begin{array} { r } { \sum _ { k \in \mathcal { O } } \mathbb { P } _ { k } = 1 , } \end{array}$ , and we assume $\beta ^ { d } < 0$ . Inspired by the model in Yang et al. (2016), we can obtain the optimal prices for each delivery choice $k$ :

Lemma 1. $\begin{array} { r } { a _ { k } ^ { \ast p r i c i n g } = C _ { k } ^ { t r u e } - r - \frac { m } { \beta ^ { d } } , ~ \forall k \in \mathcal { O } , } \end{array}$ a ∗pricing , where $a _ { k } ^ { \ast p r i c i n g }$ denotes the optimal price for delivery option $k$ and m is the unique solution to

$$
(m - 1) \exp (m) = \sum_ {k \in \mathcal {O}} \exp \left[ - \beta^ {k} \exp \left[ d _ {0, k} \right] + \beta^ {d} \left(C _ {k} ^ {\text {t r u e}} - r\right) \right], \tag {10}
$$

which we approximate using the Lambert $W _ { 0 }$ -function (Corless et al. 1996).

For all proofs, we refer to Dong, Kouvelis, and Tian (2009). Because, in reality, we do not know the cost $C _ { k } ^ { t r u e }$ of inserting a location at the moment we need to make a pricing decision, we replace $C _ { k } ^ { t r u e }$ by our approximation $\hat { C } _ { k } ^ { D S P O }$ .

# 4.2. Training the Pricing Policy

Algorithm 1 provides a high-level overview of the DSPO training procedure. The main part of the algorithm involves a simulation over I booking horizons. During this simulation, we employ the DSPO policy $\pi$ and store feature values and the accompanying observed costs. We start by obtaining a data set to train an initial neural network $\mathcal { N } _ { \theta }$ . This initial data set is obtained by running simulations without a pricing policy; that is, no incentives are provided for delivery options. After the initial training phase (step 1), we start a simulation procedure in which we simulate a horizon until cutoff time T (step 3). At each time step, we employ the DSPO pipeline: first, we obtain $a ^ { s e l e c t i o n }$ of the selection heuristic policy $\pi ^ { s e l e c t i o n }$ (step 4). Next, we obtain the pricing decision $a ^ { p r i c i n g }$ from the neural network, $\pi ^ { p r i c i n g }$ (step 5). Finally, both subdecisions are combined in a single decision $a ( s _ { t } )$ (step 6). The decision is applied to the state, the delivery choice of the customer is recorded, and the next state with the arrival of a new customer is observed (step 7). Next, the encoded state $\phi ( s _ { t } )$ is stored in a memory buffer (step 8). At the end of a simulation horizon, we can calculate the true costs $C _ { k , t } ^ { t r u e }$ related to each insertion (step 9). In experiments with finite capacity OOH locations, we add a small penalty to the cost function to steer the policy; see Appendix D. The neural network $\mathcal { N } _ { \theta }$ is trained with the Adam algorithm (Kingma and Ba 2014) (step 10). The process of data collection and updating continues until I iterations are reached.

Algorithm 1 (Training Algorithm for Dynamic Selection and Pricing of Out-of-Home Delivery)

1: Initial training phase of $\mathcal { N } _ { \theta }$   
2: for $i = 1 , 2 , \dots , I$ episodes do   
3: for $t = 1 , 2 , \dots , T$ do   
4: $a ^ { s e l e c t i o n } ( s _ { t } ) \gets \pi ^ { s e l e c t i o n } ( s _ { t } )$   
5: $a ^ { p r i c i n g } ( s _ { t } ) \gets \pi _ { \theta } ^ { p r i c i n g } ( a ^ { p r i c i n g } ( s _ { t } , a ^ { s e l e c t i o n } ( s _ { t } ) )$   
6: $a ( s _ { t } ) \gets ( a ^ { s e l e c t i o n } ( s _ { t } ) , a ^ { p r i c i n g } ( s _ { t } ) )$   
7: Apply $a ( s _ { t } )$ to state $s _ { t }$ and transition to $s _ { t + 1 }$   
8: Store encoded state $\phi ( s _ { t } )$   
9: Obtain and store the true costs of insertion true $C _ { k , t } ^ { t r u e }$ and relate them to $\phi ( s _ { t } )$   
10: Update $\mathcal { N } _ { \theta }$ and using the batch of tuples $( S , C ^ { t r u e } )$ .

# 4.3. Design Choices

A few technical comments about the design choices of our DSPO pipeline are in order. First, we base the selection policy $\mathbf { \hat { \Pi } } _ { \pi } { \hat { s e l e c t i o n } }$ on a simple heuristic rule. We argue that it is better to keep customer service high and influence customer behavior by pricing compared with limiting the offering to a smaller subset of delivery options (cf. Asdemir, Jacob, and Krishnan 2009). Therefore, we select a sufficiently large subset of delivery locations of size N. Alternatively, our machine learning model could also be used to limit the offering to customers, for example, by only offering the OOH locations that yield the lowest predicted costs. Nevertheless, we decided to use the heuristic rule as it reduces the required computations and enhances learning stability by inducing less noise in the collected data.

Our CNN architecture is relatively small compared with state-of-the-art CNNs. We found that this size of the CNN architecture allows for fast inference in a sequential training loop and requires less data to train. We noticed that larger sized CNNs did not noticeably improve the predictive performance of DSPO.

Our proposed Algorithm 1 incorporates a repeating sequence of simulation steps, data collection, and neural network updates. Hence, our implementation technically resembles the structure of reinforcement learning algorithms. The prediction model $\mathcal { N } _ { \theta }$ is generic in the sense that it can be trained by any other algorithm, for example, Deep Q-networks (Mnih et al. 2013), deterministic policy gradient (Silver et al. 2014), and proximal policy optimization (Schulman et al. 2017). We benchmark our proposed method against the latter.

# 5. Results

In this section, we discuss and analyze the performance of our policy and the benchmarks. This section is structured as follows. First, we discuss the experimental design in Section 5.1. Next, we study a small-scale synthetic case in Section 5.2 and show the results for a real case based on Amazon delivery data from the city of Seattle in Section 5.3. The synthetic case is designed to study fundamental problem characteristics, but it lacks the detailed structure found in real-world applications, such as those we examine in the Amazon case.

# 5.1. Experimental Design

We compare the DSPO policy with the following baselines and benchmark policies:

• NoOOH: the situation with only home deliveries.   
• OnlyOOH: the situation in which customers are only offered OOH delivery.   
• NoPricing: the situation in which customers can select OOH delivery if they want but do not obtain a

discount or pay a delivery charge related to their delivery choice.

• StaticPricing: the baseline that provides customers a fixed discount for choosing OOH delivery and a fixed charge for choosing home delivery.   
• Hindsight: the heuristic benchmark introduced by Yang et al. (2016) for time slot pricing and adapted to the OOH context.   
• Foresight: the improved state-of-the-art variant of the Hindsight benchmark, also proposed by Yang et al. (2016), and adapted to the OOH context.   
• Linear: instead of using the CNN, we use a linear regression model.   
• PPO: the state-of-the-art reinforcement learning actor-critic method (Schulman et al. 2017) that directly outputs prices for all delivery options.

The NoOOH and OnlyOOH baselines can be considered an approximate upper and lower bound on costs without costs or revenue from pricing. For all methods, we select only those OOH locations that still have remaining capacity. We set the number of offered OOH locations $n = 2 0$ . Note that we use $\pi ^ { s e l e c t i o n }$ for NoPricing to offer a limited set of OOH delivery locations. The StaticPricing benchmark can be considered the current situation for many retailers that offer OOH delivery and provide a static discount. The Hindsight benchmark estimates the costs (Cˆ Hink, t $( \hat { C } _ { k , t } ^ { H i n d s i g h t } )$ of inserting a location $k$ with a cheapest insertion into a preliminary route plan $R _ { t - 1 }$ . For this, we need to add a preliminary route plan $R _ { t - 1 }$ to the state; see Appendix E.1 for more details. Thcosts term $( \hat { C } _ { k , t } ^ { F o r e s i g h \bar { t } } )$ ht benchmark calculates a  of (i) the insertion costs in $R _ { t - 1 }$ hted  and RT ∈ RT: Cˆ Forek, t (ii) the avefinal routes $R _ { T } \in \mathcal { R } _ { T }$ $\hat { C } _ { k , t } ^ { F o r e s i g h t } = ( 1 - \overset { \bullet } { \boldsymbol { \theta } } _ { t } ^ { H } ) \hat { C } _ { k , t } ^ { H i n d s i g h t } + { \boldsymbol { \theta } } _ { t } ^ { H } \frac { 1 } { | \mathcal { P } | }$ $\textstyle \sum _ { p \in { \mathcal { P } } _ { - } } f ( k , p )$ , where $f ( k , p )$ represents the insertion costs of pfeasibly inserting delivery location $k$ into historic route $p$ and $\theta _ { t } ^ { \check { H } }$ is the weight. This weight decreases during the booking horizon, assuming that Cˆ Hindk,t $\hat { C } _ { k , t } ^ { H i n d s i g h t }$ becomes more accurate over time. For PPO, we use a Gaussian policy that outputs prices directly. For all pricing policies, the pricing decision space is discretized by rounding prices to two decimals; that is, the decision space size for a single delivery option $k$ is equal to $1 0 0 ( b - a )$ . Hence, the complete decision over all options $\mathcal { \kappa }$ is a vector of discrete valued numbers. In practice, one could decide on different rounding schemes based on specific requirements and constraints. We restrict prices to $[ - 1 0 , 2 ] ,$ , that is, a maximum discount of 10 and maximum delivery charge of 2 can be attributed to a delivery option. For the implementation details of the benchmarks, we refer to Appendix E.

The problem and solution methods are implemented in Python 3.10, using PyTorch 2.0.1 (Paszke et al. 2019). All vehicle routes are obtained using HGS-CVRP (Vidal et al. 2012, Vidal 2022) with the Hygese 0.0.0.8 Python

wrapper. Computations were conducted on one thin CPU node of a high-performance cluster. The node is equipped with a 2.6 GHz AMD Rome 7H12 processor and has 128 CPU cores and 160 GB of memory. All results are reported over a separate test set. For the synthetic case, the reported results correspond to averages over 30 seeds. For the Amazon case, we report results over 20 seeds.

# 5.2. Synthetic Case

We start with a study of a relatively small synthetic case to gain more structural insights into the solution and analyze and validate the performance of the proposed policies and benchmarks. First, we introduce the relevant settings in Section 5.2.1. Next, we provide insights into the cost factors in Section 5.2.2. We conduct a sensitivity analysis for multiple customer segments in Section 5.2.3 and compare DSPO with the baselines and benchmarks in Section 5.2.4, provide insights into the influence of capacities on performance and decision making in Section 5.2.5, study the decision making of DSPO when the cost distribution changes in Section 5.2.6, and end with an ablation study in Section 5.2.7.

5.2.1. Problem Setting. We utilize the Gehring and Homberger (2002) benchmark problems for data generation. Specifically, we employ the random (R), clustered (C), and random-clustered (RC) instances and partition the 200 location instances into a test and a training set. Next, we randomly select $\mid { \mathcal { L } } \mid = 1 0$ locations, which we label as OOH delivery locations with infinite capacity. The instances are illustrated in Appendix A. During data collection and evaluation, we randomly sample customer’s home locations from the respective data set. Driving times are based on Euclidean distances, and we assume a fixed vehicle speed of 30 distance units per hour. The salary costs $C ^ { \bar { w } }$ are 30 per hour, and the fuel costs $C ^ { f }$ are 0.3 per distance unit. In our experiments, we model nonuniform service durations based on spatial information. This closely resembles reality as service durations in, for instance, apartment buildings are expected to be higher compared with rural areas or suburbs. Service times at location i are obtained by projecting the service area onto the domain $( x \in [ - 3 , 3 ] , y \in$ $[ - 2 , 2 ] ,$ ) of

$$
l _ {i} (x, y) = \left(4 - 2. 1 x ^ {2} + \frac {y}{3}\right) x ^ {2} + x y + (- 4 + 4 x ^ {2}) y ^ {2}, \tag {11}
$$

which is an optimization function proposed in Molga and Smutnicki (2005); see Appendix A for a visualization of this function. We bound the service duration on $l _ { i } \in [ 1 , 1 0 ]$ minutes. The probability of delivery failure at a home address is given by $\mathbb { P } ^ { m } = 0 . 1$ with fixed costs of $C ^ { m } = 1 0$ . We draw the number of arriving customers on a day from the negative binomial distribution $D _ { \iota }$ , where

the probability mass function is given by

$$
\mathbb {P} (n; r, p) = \left(\frac {n + r - 1}{n}\right) (1 - p) ^ {n} p ^ {r}, \tag {12}
$$

where $r = 9 0 , p = 0 . 5 ,$ , and $\begin{array} { r } { \mathbb { E } [ D ] = \frac { r ( 1 - p ) } { p _ { . } } = 9 0 . } \end{array}$ . The interarp rival time of customers is uniformly distributed. The maximum number of customers served on a day is limited by the fleet capacity: we consider a limited fleet of nine vehicles with a maximum capacity of 10 parcels per vehicle. We assume a single parcel per customer. Because we do not have customer choice data, we use an MNL tuning procedure to obtain realistic choice behavior. The tuning procedure is further detailed in Appendix B. Note that we consider a single customer segment $g$ in our experiments except for the experiment in Section 5.2.3. All remaining problem parameter values are summarized in Appendix A.

5.2.2. Analysis of Cost Factors. Figure 6 shows a sensitivity analysis of the cost distribution for the RC instances. The left figure depicts the total costs (minmax scaled for visualization purposes) divided into the categories (i) travel costs, (ii) service costs, and (iii) delivery failure costs compared with the share of OOH deliveries. Travel costs are the costs related to fuel and salary $C ^ { f }$ and $C ^ { w }$ ) during travel, service costs are the salary costs during service time $l _ { i } ,$ and failure costs are the costs paid for delivery failures $( C ^ { m } )$ . The graph is obtained by using the NoPricing benchmark and tuning the MNL parameters to obtain different home delivery rates. We observe that travel costs make up the largest portion of the costs but are less sensitive to the share of OOH deliveries. The travel costs only slightly increase from $0 \%$ home delivery to $2 0 \%$ home delivery but thereafter stay relatively stable. All areas are visited by vehicles, so the travel distances do not significantly increase as the number of stops in the area increases with more home deliveries. The service and delivery failure costs are elastic; that is, as more customers select home delivery, these costs increase significantly.

Observation 1. Travel, service, and delivery failure costs rise with home deliveries; OOH delivery mainly impacts costs through reduced service durations and delivery failures.

The right graph in Figure 6 illustrates the total operational costs, including travel, service, and delivery failure expenses, along with the costs incurred from offering fixed discounts in situations in which no delivery charge is applied. The graph shows that (i) providing discounts can significantly reduce operational costs and (ii) a balance needs to be struck when offering discounts, making sure they do not harm the overall profitability. The concept of finding a balance aligns with the findings from Lin et al. (2022), who

![](images/685cbeb29f857b992d05f4c46e974cf8d8d8bc8c0661d990f097e7c2b0dd4011.jpg)  
Figure 6. Sensitivity Analysis of Costs and Discounts, Results Based on the RC Instances

![](images/eada4664bd7db2c23a2bc82af98e12fbd4c99d09215cb4d52edcaa8602107243.jpg)

examine the optimal number of OOH locations to open. Overall, the total costs seem approximately convex in the discount percentage and the minimum of costs lies at $4 \%$ fixed discounts as a percentage of revenue per customer r. Considering the approximate convexity, we note that the determined discount value is optimal within the fixed discount policy and does not imply a globally optimal policy.

Observation 2. Offering discounts can substantially reduce operational costs, but they might compromise overall profitability.

Next, we study the effect on travel costs when using the random, clustered, and RC instances of Gehring and Homberger (2002). Figure 7 shows boxplots for the R, C, and RC instances (left to right). For each instance type, we show the effect of $0 \%$ , $5 0 \%$ , and $1 0 0 \%$ of the customers choosing home instead of OOH delivery. The results confirm the observations from Figure 6: the travel costs increase from $0 \%$ to

$5 0 \%$ , but thereafter, the costs hardly increase. We do not find significant differences in travel costs between the R, C, and RC instances. Therefore, for the remaining experiments, we focus solely on the RC instances.

Observation 3. Only offering OOH delivery (no home deliveries) leads to significant travel cost savings across spatial dispersion types (random, clustered, and random clustered).

5.2.3. Analysis of Customer Segments. In our experiments, we so far considered a single customer segment g. This means that all customers are willing to accept OOH delivery, which might not be realistic. When more customer choice data becomes available for research, one may define distinctive customer segments as we already modeled in Section 3.2. In this section, we study the effects of modeling three customer segments: customers who (i) only consider home delivery and are nonsensitive to incentives, (ii) consider both OOH and

![](images/e2e34cb4406bdaf1e8f410bf1f0f14ae168f594d08305a8097ef68b9bc321783.jpg)  
Figure 7. Travel Costs for Different Spatial Distributions and OOH Shares Given $C ^ { f } = 2 . 0$ over 30 Replications

home delivery and are somewhat sensitive to both the incentives $( \beta ^ { d } )$ and the required distance to travel to an OOH location $( { \boldsymbol { \beta } } ^ { k } ) _ { \cdot }$ , and (iii) consider both OOH and home delivery and are highly sensitive to incentives $( \beta ^ { d } )$ and less sensitive to distance $( { \boldsymbol { \beta } } ^ { k } )$ . We define the customer choice model parameters per segment in Appendix B. Figure 8 shows a sensitivity analysis of the shares of the different customer segments. We vary the parameters $\mu _ { 1 } , \mu _ { 2 } ,$ and $\mu _ { 3 } ,$ which define the probability that an arrived customer is from segment 1, 2, or 3, respectively. The different lines in the graph represent different levels of $\mu _ { 1 } ,$ and the $x$ -axis of the left and right graph define the value for $\mu _ { 2 }$ . The value of $\mu _ { 3 }$ is implicit, that is, $\mu _ { 3 } = 1 . 0 - \mu _ { 1 } - \mu _ { 2 }$ . The results are shown for the RC instance, using the StaticPricing benchmark and infinite OOH capacity. Note that we defined different static pricing levels for each setting, aimed at finding the lowest costs during simulation. On the left, we show the percentage of customers choosing home delivery. This follows a logical pattern as a larger share of segment 1 customers indicates more home deliveries and vice versa for segments 1 and 2. The right graph shows the total costs (operational $^ +$ pricing costs) for the different values of $\mu _ { 1 } , \ \mu _ { 2 } ,$ and $\mu _ { 3 }$ . Remember that $\mu _ { 1 }$ indicates only home delivery, $\mu _ { 2 }$ is open to both home and OOH delivery, and $\mu _ { 3 }$ prefers OOH delivery. We observe that a higher share of customers that select OOH delivery results in lower costs. Moreover, the cost reduction seems to show only a slightly increasing effect in the OOH deliveries; that is, a higher share of customers choosing home delivery $( \mu _ { 1 }$ increase) also results in higher costs. We conjecture that this effect is caused by the economies of scale by bundling more deliveries into OOH locations. Considering the share of segment 2 and 3 customers, we see that it is easier to nudge segment 3

customers to OOH delivery, resulting in more OOH deliveries and lower costs from pricing.

Observation 4. The delivery costs decrease more strongly when a higher share of customers is sensitive to incentives for choosing OOH delivery.

For the remainder of the experiments, we use a single customer segment again and rely on our tuning procedure detailed in Appendix B.

5.2.4. Comparison of DSPO with Benchmarks. Table 2 shows the results on the synthetic case for all benchmarks and DSPO. We show the relative savings compared with the NoOOH baseline and the $9 5 \%$ confidence interval of these savings. The difference in costs (27.2 percentage points $( \% \mathrm { { p t } ) }$ ) between NoOOH and OnlyOOH indicates an approximate upper bound on the savings a policy can yield. Note, however, that OnlyOOH does not incur costs for providing discounts as all other policies do. The StaticPricing benchmark provides a static discount (five) for choosing OOH delivery and a static delivery charge (two) for home delivery for each customer. These values were selected after an experimental evaluation aimed at finding the lowest costs during simulation. Even using StaticPricing, $7 . 9 \%$ in total costs can be saved compared with not offering OOH and 3.9%pt in costs can be saved compared with the NoPricing benchmark. This shows that even StaticPricing can already save significantly in costs. The Hindsight benchmark does not convincingly beat the StaticPricing benchmark. The Hindsight benchmark overestimates the costs of delivery and provides too high discounts. The Hindsight benchmark is solely basing its pricing on the preliminary route plan $R _ { t - 1 } ,$ which is especially inaccurate at the beginning of the

![](images/d788b99062e1446ac3175913cb776de959b07c352a62ce9d4a71c2ccc72ccddd.jpg)  
Figure 8. Analysis of the Customer Segment Division, Results Based on the RC Instances Using StaticPricing

![](images/f38c485b52d86149b0b7daabd77310967e6502df857a422ada0dcb4b2ee29be3.jpg)

Table 2. Results on the Synthetic Case, Reported on the RC Instances   

<table><tr><td></td><td>Percentage home delivery</td><td>Travel costs</td><td>Service costs</td><td>Delivery failure costs</td><td>Discount costs (charge revenue)</td><td>Average discount (average charge)</td><td>Percentage savings (95% confidence interval)</td></tr><tr><td>NoOOH</td><td>100</td><td>2,353.7</td><td>523.3</td><td>89.0</td><td>—</td><td>—</td><td>—</td></tr><tr><td>OnlyOOH</td><td>0</td><td>2,145.1</td><td>15</td><td>0.0</td><td>—</td><td>—</td><td>27.2 (±0.2%)</td></tr><tr><td>NoPricing</td><td>81.7</td><td>2,346.1</td><td>427.5</td><td>72.7</td><td>—</td><td>—</td><td>4.0 (±0.1%)</td></tr><tr><td>StaticPricing</td><td>59.1</td><td>2,331.3</td><td>296.9</td><td>50.5</td><td>159.3 (107.0)</td><td>5 ± 0 (2 ± 0)</td><td>7.9 (±0.2%)</td></tr><tr><td>Hindsight benchmark</td><td>40.9</td><td>2,257.5</td><td>205.2</td><td>34.9</td><td>386.0 (87.3)</td><td>9.7 ± 1.5 (2.0 ± 0.1)</td><td>5.7 (±0.3%)</td></tr><tr><td>Foresight benchmark</td><td>45.6</td><td>2,238.5</td><td>229.1</td><td>39.0</td><td>320.3 (104.5)</td><td>9.4 ± 2.0 (2.0 ± 0.2)</td><td>8.2 (±0.2%)</td></tr><tr><td>Linear benchmark</td><td>70.4</td><td>2,265.7</td><td>354.8</td><td>60.3</td><td>190.4 (95.8)</td><td>1.8 ± 0.4 (5.6 ± 2.8)</td><td>6.4 (±0.3%)</td></tr><tr><td>PPO benchmark</td><td>61.7</td><td>2,329.7</td><td>308.5</td><td>52.5</td><td>172.5 (52.7)</td><td>3.6 ± 2.5 (1.4 ± 0.7)</td><td>5.2 (±0.2%)</td></tr><tr><td>DSPO</td><td>70.4</td><td>2,196.2</td><td>341.0</td><td>58.0</td><td>192.4 (84.3)</td><td>5.2 ± 3.7 (1.8 ± 0.5)</td><td>8.8 (±0.2%)</td></tr></table>

booking horizon. The Foresight benchmark, as proposed by Yang et al. (2016), tries to overcome this by using a pool of historic routes $\mathcal { R } _ { T }$ to determine costs. However, the accuracy of the prediction based on historical route data is constrained by variability in daily customer arrivals, which follow a negative binomial distribution D. As with the Hindsight benchmark, this also results in an overestimation of the costs although the Foresight benchmark does outperform the Static-Pricing benchmark, which is in line with the results obtained by Yang et al. (2016).

Observation 5. Cheapest insertion as a cost predictor (Hindsight benchmark) overestimates home delivery costs, leading to excessive discounts. This overestimation is only partially compensated by incorporating historic routes (Foresight benchmark).

Next, we compare three learning approaches: the Linear benchmark, PPO, and DSPO. We observe that the Linear benchmark is unable to differentiate between customers in terms of costs, which is reflected by the low deviation in discounts and charges. Interestingly, both PPO and DSPO find a policy that can differentiate between customers as reflected by the deviation of discounts. However, DSPO outperforms PPO mainly because of lower travel costs and slightly higher revenue from delivery charges. Even though the use of PPO results in fewer delivery stops, DSPO can recognize and nudge remote customers better to OOH options, which yields lower travel costs. We conjecture that PPO learns a policy that differentiates between customers but does not always provide discounts to the right customers. Note that PPO requires one million episodes to learn a performant policy because of the sparse cost structure of our problem; see Appendix E.3 for the convergence curve. The convergence curve of DSPO, which only requires 100,000 episodes, is depicted in Appendix D.

Observation 6. DSPO can differentiate between customers, identify the customers that are most expensive for home delivery, and adjust pricing accordingly.

5.2.5. Analysis of Finite Capacity OOH. So far, we have assumed infinite capacity of all OOH locations. However, in reality, many OOH locations have a limited capacity; see Sethuraman et al. (2024). In this section, we study the effect of finite capacity OOH locations. Furthermore, we study the effect of different capacities per OOH location.

First, we show the general effect of a capacitated system on the performance of the Foresight benchmark, PPO, and DSPO; see Figure 9. We randomly assign OOH locations to have limited capacity, varying the capacity of capacitated OOH locations between $\{ 2 , 3 , 5 \}$ and the fraction of capacitated OOH locations varying between $\{ 0 \% , 3 0 \% , 5 0 \% , 8 0 \% , 1 0 0 \% \}$ . Note that DSPO and PPO are retrained for every setting. The Foresight benchmark is unable to cope with the finite capacity system as it does not differentiate its pricing between OOH locations depending on remaining capacity. Hence, the costs for the situation with $1 0 0 \%$ finite capacity OOH locations and low capacity (left) are higher than the NoOOH baseline caused by the added costs of providing discounts. PPO, being a learning method that can utilize information considering the remaining OOH capacities, is better able to cope with the capacitated systems. In all settings, it incurs a slight increase in costs when the share of capacitated OOH locations is increased. DSPO has a similar pattern in cost increase across the experimental settings. Nonetheless, the overall advantage of DSPO compared with PPO remains consistent with the earlier results from Table 2. We conjecture that DSPO’s superior performance in a capacitated context is due to its underlying CNN architecture being better able to deal with this added complexity. DSPO effectively deters customers with lower delivery costs from choosing OOH delivery, encouraging them to opt for home delivery instead. This strategy preserves future capacity for more expensive customer deliveries, resulting in savings on routing costs. We establish that the finite capacity problem is inherently more challenging as the pricing policy needs to adapt, for example, by nudging a customer to a less preferred option as the most preferred option has no remaining capacity.

![](images/ec4cf2f6df713ce8568fc7b4c394aff57b3bf524a5802531a89fbd3f2d9dec64.jpg)  
Figure 9. The Total Costs Incurred by Foresight Benchmark, PPO, and DSPO for Different Fractions of Capacitated OOH Locations (X-Axis) with Low Capacity (Left), Medium Capacity (Middle Column), and High Capacity (Right)

![](images/163bc2cb5f0b0c116800a0f4fe7230cd0fcbe3c773e81091d8def1c2f2df654c.jpg)

![](images/0cf801b82b67693416e6f0ccbbbbc75b6cfd7647c0e5df268f72106165044919.jpg)

Observation 7. Finite capacity at OOH locations increases costs, and DSPO consistently outperforms both the Foresight benchmark and PPO by better managing capacity.

Next, we visualize the decision making of PPO and DSPO during a booking horizon in a capacitated system with each location having a capacity of three; see Figure 10. We intentionally study a system with limited capacity to amplify their effects. In the top row, we show the remaining total OOH capacity during a booking horizon (collected over 1,000 episodes). The figure confirms that PPO is less able to preserve capacity for later arriving customers compared with DSPO. This way, DSPO can provide OOH delivery longer during the booking horizon, potentially saving capacity for customers that are relatively more expensive to serve at home. However, both policies make sure that OOH locations are fully exploited. Note that, over the 1,000 replications, the capacity of PPO reduces approximately linearly during the booking horizon, but in single episodes, still approximately $6 0 \%$ of the customers select home delivery. In the bottom row, we display the average accepted discount for OOH delivery over the same booking horizon. The bars are aggregated over five time steps. The figure illustrates that PPO significantly adjusts its incentives only when there is no remaining capacity. In contrast, DSPO offers lower incentives at the beginning of the booking horizon to deter customers from choosing OOH delivery, thereby preserving some capacity. Later, DSPO provides higher incentives as it gains more certainty about future customer arrivals, effectively nudging more customers toward OOH delivery.

Observation 8. DSPO better preserves OOH capacity than PPO, offering lower initial incentives to deter OOH delivery early and higher incentives later to manage capacity effectively.

5.2.6. Sensitivity of DSPO to Fuel and Salary Costs and Service Time. Next, we show how DSPO can adjust its policy to different levels of fuel and salary costs $C ^ { f }$ and $C ^ { w }$ ). Figure 11 depicts the percentage of home delivery for different settings. From left to right, the salary costs increase from 30 to 50 per hour, and from top to bottom, the costs of fuel increase from 0.6 to 2.0 per distance unit. On the $x$ -axis of each subfigure, we increase the upper bound on service time $l _ { i }$ . Note that DSPO is retrained for every setting. The left column shows that higher fuel costs have a modest effect on the percentage of customers who are nudged toward OOH delivery. In contrast, higher salary costs have a bigger impact as DSPO decides to nudge more customers to OOH delivery options as the service times $l _ { i }$ increase.

Observation 9. When salary costs increase and the service times are high, DSPO adapts by giving higher discounts and nudging more customers to OOH delivery.

5.2.7. Ablation Study for DSPO. To obtain a better understanding of the working of DSPO, we conduct three ablations, whose results are summarized in Figure 12. The first ablation is the removal of the CNN network structure; that is, the matrix state representation is flattened and directly fed to the FC layers, similar to the state as fed to the Linear benchmark; see Appendix E. This way, we show the added value of the feature extraction layers of the CNN. Without the convolutional layers, the total costs increase by $1 4 . 8 \%$ compared with the nonablated DSPO. This shows that the CNN layers help in extracting information from the encoded state. Without this feature extraction, the FC layers are less able to find a performant policy. The second ablation is done on the training algorithm. Instead of updating the weights of the neural network after every iteration (step 10 in Algorithm 1), we only give it a one-shot

![](images/7b88088ec9232c54f2caeffca56d877162b8540173ba9489a8c27e7a6f9505df.jpg)  
Figure 10. The Remaining Total Capacity (Top) and Accepted Aggregated OOH Discounts (Bottom) During the Booking Horizon for a System with Only Capacitated OOH Locations with Capacity of Three; Data Collected over 1,000 Episodes with a Shaded Area of Two Standard Deviations

![](images/f1345761f15f5f9ad20cb22a1295b63e829f8e4a9267743ef7aa9d9e3fa461ae.jpg)

![](images/513007c80b48fb308b93920cae5c7f882f17b1a3516d552e8060ea76437afc73.jpg)

![](images/28f273d1c4060197e6743cfda7dfcf5c4d3a1dc68abed9ea94b4bfe86a8007c9.jpg)

opportunity by providing it with a data set to train on once without the opportunity to retrain on newer observations; that is, we only conduct the initial training phase (step 1 in Algorithm 1). We find that not retraining DSPO increases costs by $4 . 2 \%$ . This shows that retraining on new observations is important and motivates the retraining in a reinforcement learning fashion. Finally, we show the effect of doing both ablations simultaneously. Not surprisingly, this yields worse results, incurring $1 5 . 7 \%$ more costs compared with nonablated DSPO.

Observation 10. The utilization of convolutional neural networks for state representation, coupled with iterative training loops, is a powerful combination enhancing predictive performance.

# 5.3. Amazon Case

In this section, we show the results for a real-world inspired case. First, we explain the problem setting in Section 5.3.1. Next, we compare DSPO with the benchmarks in Section 5.3.2 and analyze the pricing decisions of DSPO and compare it against the best performing benchmark in Section 5.3.3.

5.3.1. Problem Setting. We study a real-world case using the order data of the large retailer Amazon in the city of Seattle. We use publicly available data of Amazon from the greater Seattle area (Merchan et al. 2022). Amazon offers OOH delivery in Seattle, but these locations are not contained in the publicly available data set. Hence, we built a tool that scrapes the Amazon site for OOH locations; see Amazon (2024a). The data set with the customer and OOH locations is illustrated in Figure 13.

For each arriving customer, we draw a customer location from the data set. We calculate the driving distances and times using the real road network, including congestion on an average day, obtained from ArcGIS 10.8 (Redlands ESRI 2024). Again, we draw the service times $l _ { i }$ from Equation (11) with as bound $l _ { i } \in [ 1 , 1 0 ]$ . The number of arriving customers on a day is drawn from the negative binomial distribution $D$ with $r = 7 0 0$ and $p =$ 0.5, which yields $\mathbb { E } [ D ] = 7 0 0$ . We consider uniformly distributed interarrival times for each customer. Each vehicle in the fleet of 25 vehicles has a capacity of 100 packages. The fleet operates from a single central depot. Note that the fleet capacity is much higher than the expected number of customers on a day (700). This

![](images/64eb58bcd7150c9e7d71546631347868e0e638ba7a652a51451452d04317f9e6.jpg)  
Figure 11. The Percentage of Home Deliveries of DSPO for Normal Fuel Costs (Top) and High Fuel Costs (Bottom) Given Normal Salary Costs (Left) and High Salary Costs (Right) for Different Levels of the Maximum Service Time

![](images/c82b67a6e9ac26d1f7885b7f8536f575d1d3c849a500b3d8f8997879bb438e41.jpg)

![](images/c3eaa9af9886af52a41a301111a09b245a8ff9f89ad443beb8a3471e81c21746.jpg)

![](images/ac3d53ac085815e41f212fad210119c88e553f2e315f04640456acc4678cd234.jpg)

allows us to cope with possible busy days, that is, when we draw $\gg 7 0 0$ from D. The data set contains $\mid { \mathcal { L } } \mid = 2 9 9$ OOH locations. In the data, $3 8 \%$ of the OOH locations are capacitated with each having a capacity of 42 parcels (cf. Amazon 2024b). The costs of salary, fuel, and delivery failure are $C ^ { f } = 0 . 3 , C ^ { w } = 3 0$ , and $C ^ { m } = 1 0$ , respectively. The probability of delivery failure is $\mathbb { P } ^ { m } = 0 . 1$ .

5.3.2. Comparison of DSPO with Benchmarks. Table 3 shows the results of DSPO and the benchmarks on the Seattle case. We observe that, compared with the synthetic case, overall differences between policies are larger. Because we utilize actual road network driving times, the logistic cost benefits of customers opting for OOH delivery are amplified, particularly for those in remote or traffic-dense areas, which aligns with the results from Janinhoff, Klein, and Scholz (2023). By providing customers the option of OOH delivery without offering discounts (NoPricing), $9 . 8 \%$ in costs can be saved compared with only offering home delivery (NoOOH). The StaticPricing benchmark offers discounts of five for choosing OOH delivery and a delivery charge of two for home delivery and can save even more $( 1 3 . 8 \% )$ by nudging a large number of customers to OOH delivery. The Hindsight and Foresight

benchmarks outperform the StaticPricing benchmark: both find a policy that nudges many customers to OOH delivery. Compared with the StaticPricing benchmark, both policies save mainly on travel and service costs. The performance of the learning policies differs a lot: the Linear benchmark and PPO are unable to find a performant policy, whereas DSPO finds the best overall policy, saving $2 0 . 8 \%$ in total costs compared with NoOOH. Clearly, the Linear benchmark is unable to abstract the complex relationships from the encoded state, and hence, it does not yield an accurate estimation of the costs. Even after significant tuning effort, PPO does not converge within 600,000 episodes. We conjecture

![](images/f80531edd173eb060e4dc519170091b6e804fefdf22073f71101d62c5559c247.jpg)  
Figure 12. Bar Chart Showing the Total Costs for Different Ablations of DSPO

![](images/ed2b3cd299f6cd2a4cb4e5f93a585f42d59ebb4733d72684ca2ade5a20bda18f.jpg)  
Figure 13. (Color online) Seattle Instance Map with Customer and OOH Locations (North Oriented to the Right)   
Customer OOH delivery location

that PPO is unable to deal with the sparse cost and long episode length as encountered with this problem: only at the cutoff time $T$ are the costs revealed, which troubles learning convergence. All convergence curves can be found in Appendices D and E. DSPO, however, can find a performant policy that nudges almost half of the customers to OOH delivery. It seems that, compared with the synthetic case, it is more profitable to nudge many customers to OOH delivery, whereas for the synthetic case, it was sometimes better to provide lower discounts because home delivery was not always as expensive. Compared with the state-of-the-art benchmarks, DSPO saves $1 9 . 9 \% \mathrm { p t }$ (compared with PPO) and $3 . 8 \% \mathrm { p t }$ (compared with the Foresight benchmark) in total costs. Because of the limited number of finite capacity OOH locations $( 3 8 \% )$ and their fairly high capacity, no OOH location reaches its capacity on a single delivery day.

Observation 11. DSPO finds the best policy and saves 19.9%pt in total cost compared with not offering OOH delivery, 7%pt compared with a static pricing policy, and 3.8%pt compared with the best performing stateof-the-art benchmark (Foresight benchmark) for the Amazon case.

5.3.3. Analysis of Pricing Decisions. Figure 14 depicts the accepted discounts for OOH delivery during a booking horizon for the Foresight benchmark (left) and DSPO (right). The color and size of the scatter points indicate the distance from the customer’s home to the OOH location. Results are plotted over a single, exemplary booking horizon. At the beginning of the booking horizon, the Foresight benchmark seems to provide similar discounts to all customers. As time progresses and the booking horizon advances, the estimated cost for a new customer increases, leading to higher discounts (in

the time interval from 600 to 800). Only toward the end of the horizon does the cost estimation return to the level of the beginning of the horizon, providing similar discounts. Comparing this pricing behavior with DSPO, we observe three major differences: (i) DSPO has a slightly higher deviation in accepted discounts, (ii) DSPO provides higher discounts to customers that have to travel further for an OOH location, and (iii) DSPO seems to provide the highest discounts in the time interval from 200 to 600 of the booking horizon.

Observation 12. DSPO can adapt to customer behavior by providing higher discounts to customers who are less likely to select OOH delivery.

It seems that the 200–600 interval of the booking horizon is most crucial in nudging customers. DSPO seems to anticipate customer arrivals and customer behavior by giving higher discounts to remote customers early in the booking horizon. After this interval, nudging customers is of less importance to DSPO as it provides lower discounts. Only at the end of the booking horizon does DSPO provide higher discounts again.

Observation 13. DSPO can adapt its pricing policy to the time of arrival in the booking horizon. Specifically, the interval between $2 0 \%$ and $6 0 \%$ of the elapsed booking horizon is critical for effectively nudging customers.

# 6. Conclusion

In this paper, we study the dynamic selection and pricing of OOH deliveries. The studied problem is novel because it considers (i) stochastic customer arrivals, (ii) stochastic customer choice, and (iii) dynamic decision making as we make sequential decisions on the customer incentives provided for OOH delivery without

Table 3. Results on the Amazon Seattle Case   

<table><tr><td></td><td>Percentage home delivery</td><td>Travel costs</td><td>Service costs</td><td>Delivery failure costs</td><td>Discount costs (charge revenue)</td><td>Average discount (average charge)</td><td>Percentage savings (95% confidence interval)</td></tr><tr><td>NoOOH</td><td>100</td><td>2,091.4</td><td>2,948.1</td><td>839.5</td><td>—</td><td>—</td><td>—</td></tr><tr><td>OnlyOOH</td><td>0</td><td>1,404.7</td><td>378.0</td><td>0.0</td><td>—</td><td>—</td><td>69.7 (±0.7%)</td></tr><tr><td>NoPricing</td><td>81.4</td><td>2,221.9</td><td>2,400.5</td><td>683.5</td><td>—</td><td>—</td><td>9.8 (±0.9%)</td></tr><tr><td>StaticPricing</td><td>56.3</td><td>2,178.5</td><td>1,658.9</td><td>472.4</td><td>1,738.5 (983.5)</td><td>5±0 (2±0)</td><td>13.8 (±1.0%)</td></tr><tr><td>Hindsight benchmark</td><td>53.3</td><td>2,099.0</td><td>1,461.0</td><td>416.0</td><td>1,830.6 860.0)</td><td>5.2±0.6 (2±0)</td><td>15.9 (±1.2%)</td></tr><tr><td>Foresight benchmark</td><td>54.2</td><td>2,075.8</td><td>1,485.6</td><td>423.0</td><td>1,767.5 (876.7)</td><td>5.2±1.2 (2±0.1)</td><td>17.0 (±1.1%)</td></tr><tr><td>Linear benchmark</td><td>62.8</td><td>2,176.6</td><td>1,914.0</td><td>545.0</td><td>1,610.9 (806.5)</td><td>4.0±2.1 (1.7±0.5)</td><td>7.5 (±1.2%)</td></tr><tr><td>PPO benchmark</td><td>60.4</td><td>2,193.5</td><td>1,784.5</td><td>508.1</td><td>1,820.3 (478.9)</td><td>3.6±2.5 (1.4±0.7)</td><td>0.9 (±1.4%)</td></tr><tr><td>DSPO</td><td>52.8</td><td>2,068.3</td><td>1,311.8</td><td>409.2</td><td>1,748.8 (881.4)</td><td>5.1±2.3 (2±0.1)</td><td>20.8 (±1.3%)</td></tr></table>

knowing the customers that will arrive in the remainder of the day. We define an MDP for the studied problem, present a novel solution, and study a small synthetic case before moving to a real-world inspired case of deliveries in Seattle. We propose DSPO, an algorithmic pipeline that uses a novel spatial-temporal state encoding and a CNN to estimate the costs of delivery. DSPO subsequently determines optimal prices for a selected subset of OOH locations. We compared DSPO with two state-of-the-art benchmarks: a method adapted from time slot demand management literature and PPO.

Our insights have significant implications for both theory and practice in the field of last-mile logistics. We show that, in a dynamic setting, the savings from offering OOH delivery are mainly the result of shorter service times and lower delivery failure rates. Furthermore, we show that using incentives for choosing OOH locations is effective: static delivery charges for home delivery and static discounts for OOH delivery can already save $4 . 0 \%$ in total costs compared with a situation without delivery incentives. However, a balance should be struck as offering too much discount can potentially harm overall profitability. Our DSPO pipeline can save 19.9%pt in total cost compared with not

offering OOH delivery and 7%pt in total costs when compared with a static pricing policy. Compared with the state-of-the-art benchmarks, we can save from 3.8%pt up to $1 9 . 9 \% \mathrm { p t }$ for the Seattle case study. Understanding the nuances of customer behavior concerning OOH delivery choices, particularly in areas with varying OOH location densities and OOH capacities, is vital. Managers can leverage our insights to tailor pricing strategies effectively, optimizing operational efficiency and maintaining high levels of customer satisfaction. The limitation of our findings lies mainly in the absence of customer choice data. We assume customers to be mainly influenced by the distance they need to travel to an OOH location from their home address although this might not be accurate. This remains a topic for further research.

Many more avenues for further research remain unexplored. One area of interest involves problem extensions and variants to the studied problem, including scenarios with heterogeneous parcel lockers with differing capacities; time windows for home delivery; more refined customer segmentation, for example, different preferences between automated parcel lockers (open 24/7) and staffed shops (limited opening hours); shifting failed

![](images/6edd944b122da5369e1ece625afa078dc6549ad49c1cca17e21231be4e883486.jpg)  
Figure 14. Analysis of the Accepted OOH Discounts over the Booking Horizon Given the Distance from the Customer’s Home to the OOH Location, Moving Average over 20 Time Steps

![](images/4e18f799c3284d7ceba1bb9f08a0f99c5e5d58e6f376033144ce269d30958478.jpg)

![](images/de30fdbf5da5cd7b90e03c387e0242552734a50e2f5df2735ff02b82010f07c4.jpg)

home deliveries to OOH locations; a multiday planning horizon, including finite capacity and package dwell time; a combination of dynamic offering and pricing; and the possibility of autonomously driving parcel lockers. In general, our problem motivates different pricing schemes based on both spatial and temporal differences between customers, which might be interesting for further research. Our state-encoding could be further refined, for example, considering an automated way to use knowledge of the spatial and arrival time distributions to define better spatial and temporal aggregations using clustering algorithms. Additionally, studying the cost sharing between retailers and third-party logistics

# Appendix A. Problem Settings

In this section, we provide all information related to the problem settings and problem parameters of the synthetic case and the Amazon case. All parameter settings are summarized in Table A.1.

Figure A.1 illustrates the domain of the six-hump camelback function used for both the synthetic and Amazon

providers highlights a complex arrangement: retailers manage discounts and delivery charges to guide customer behavior and aim to reduce logistics costs, but the logistics providers incur these operational expenses.

# Acknowledgments

F. Akkerman conducted his research in the project Dyna-Plex: Deep Reinforcement Learning for Data-Driven Logistics, made possible by TKI Dinalog and the Topsector Logistics and funded by the Ministry of Economic Affairs and Climate Policy of the Netherlands. The authors thank Jana Finkeldei and Alexander Reger for their help in obtaining the Seattle OOH data.

case to obtain service times $l _ { i }$ per location i. The service area is projected onto this function domain (see Equation (11)) to obtain service durations. We ensure that service times are not too low or high by clipping, $l _ { i } \in [ 1 , 1 0 ]$ .

Figure A.2 illustrates an RC instance as used from Gehring and Homberger (2002). Here, we indicate the OOH

Table A.1. Problem Settings   

<table><tr><td></td><td>Cw</td><td>Cf</td><td>Cm</td><td>r</td><td>li</td><td>Pm</td><td>V</td><td>K</td><td>|L|</td><td>a pricing ∈ [a,b]</td></tr><tr><td>Synthetic case</td><td>30</td><td>0.3</td><td>10</td><td>50</td><td>Equation (11)</td><td>0.1</td><td>9</td><td>10</td><td>10</td><td>[−10,2]</td></tr><tr><td>Amazon case</td><td>30</td><td>0.3</td><td>10</td><td>50</td><td>Equation (11)</td><td>0.1</td><td>25</td><td>100</td><td>299</td><td>[−10,2]</td></tr></table>

![](images/4dd76368898fcec9216752a3e96c011ef7353c3c576783a1da1e161f835c1878.jpg)  
Figure A.1. Six-Hump Camel Function for Obtaining Service Times per Spatial Area (Molga and Smutnicki 2005)

Figure A.2. Gehring and Homberger (2002) RC Instance Map with Customer and OOH Locations   
![](images/6db25fd4f3357932cf2176b15947d801d3bc4ab3e7595332f90112f25d71dc7f.jpg)  
Customer OOHdelivery location

delivery locations in blue (larger dots) and customer locations in gray (smaller dots).

# Appendix B. Choice Model Tuning

We denote all MNL parameters without customer segments. Because we do not have detailed customer segment data, we use a tuning procedure to obtain realistic behavior with a single segment. Note that we do conduct a sensitivity analysis in which we do consider multiple segments; see Section 5.2.3. For the other experiments, we used this tuning procedure and assume a single customer segment.

The utility of home delivery, $u _ { k } ,$ with $k = h ,$ is always the same if we use Equation (5). Therefore, we replace it with a separate utility term for home delivery, denoted by $u _ { 0 } ^ { + }$ . We use a simple iterative procedure in which we tune three parameters: the utility of home delivery, $u _ { 0 } ^ { + } ,$ and the sensitivity to distance and pricing, $\beta ^ { k }$ and $\beta ^ { d }$ , respectively. Our data on customer preferences is limited; however, it includes information on the proportion of OOH deliveries in countries where OOH providers are well-established as

Table B.1. Multinomial Choice Model Parameter Settings for a Single Segment   

<table><tr><td></td><td>βk</td><td>u0+</td><td>βd</td><td>Gumbel μ</td><td>Gumbel β</td></tr><tr><td>Synthetic case</td><td>0.02</td><td>3.2</td><td>-0.25</td><td>0</td><td>1.0</td></tr><tr><td>Amazon case</td><td>0.018</td><td>3.55</td><td>-0.18</td><td>0</td><td>1.0</td></tr></table>

well as the OOH market share in countries where OOH services are still gaining adoption; see, for instance, Loquate (2021) and Last Mile Experts (2022). For $u _ { 0 } ^ { + } , \beta ^ { k } ,$ , and $\beta ^ { d }$ , we set a range of possible values $\left[ - 5 . 0 , 5 . 0 \right]$ with 0.01 increments. First, we tune the utility of home delivery using a simulation of 100 replications in which all possible values for $u _ { 0 } ^ { + }$ are tested. During this simulation, we use the NoPricing baseline. We store the percentage of customers selecting home delivery and next choose the value of $u _ { 0 } ^ { + }$ and $\beta ^ { k }$ that yield (close to) $8 0 \%$ of the customers choosing home delivery. Next, we use the StaticPricing benchmark and select the values that yields (close to) $6 0 \%$ home deliveries. The $8 0 \%$ and $6 0 \%$ rates of home deliveries correspond to countries where OOH delivery options are gaining popularity and where OOH delivery is already well-established, respectively. All MNL parameter values are summarized in Table B.1.

In Section 5.2.3, we conduct a sensitivity analysis using three segments g. In Table B.2, we define the relevant MNL parameters for the three segments. Segment 1 only considers home delivery, so it is not modeled using the MNL parameters.

Table B.2. Multinomial Choice Model Parameter Settings for Three Segments   

<table><tr><td></td><td>βk</td><td>u0+</td><td>βd</td><td>Gumbel μ</td><td>Gumbel β</td></tr><tr><td>Segment 1</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>Segment 2</td><td>0.015</td><td>3.1</td><td>-0.18</td><td>0</td><td>1.0</td></tr><tr><td>Segment 3</td><td>0.05</td><td>1.0</td><td>-0.35</td><td>0</td><td>1.0</td></tr></table>

![](images/2c48a72bdf9fa3ca4c8e0bf9830766f9bfc542baa4e7f4ce5664c92548c39d48.jpg)  
Figure C.1. Sensitivity Analysis of the MNL Choice Model, Results Based on the RC Instances

![](images/4a0adaaa16053bff4f44a286eab589f9e316713bcd1990e8512180e657b16b26.jpg)

# Appendix C. Choice Model Validation

In this section, we validate the choice model tuning procedure by showing the effect of pricing on customer choice. All graphs in this section are shown for the single segment model, which is used for most experiments. Figure C.1 shows a sensitivity analysis of the MNL choice model. The left graph shows the percentage of customers choosing home delivery in relation to the relative OOH density. The relative OOH density is calculated using the number of OOH locations divided by the number of customers in the service region. We do not provide discounts in this setting. The graph clearly shows that more customers choose OOH delivery as the OOH density increases. However, as the OOH density approaches $1 2 \%$ , the area is saturated, and the percentage of home deliveries only decreases slightly thereafter. This aligns with the results in Enthoven et al. (2020). Note that our choice model calculates utility based on the distance from customer homes to OOH locations.

The right graph of Figure C.1 shows the percentage of home delivery for different fixed discounts as a percentage of revenue. Each line in the graph represents different

home delivery charges as a percentage of revenue, that is, the costs that a customer has to pay when choosing for home delivery. The graph validates that our MNL model is sensitive to discounts and charges.

Finally, we analyze the effect of discounts and OOH density on the willingness of the modeled customers to accept an OOH delivery further away from their homes. Figure C.2 shows how far customers are willing to travel given OOH density $( y$ -axis) and fixed discounts (x-axis). A darker color indicates a further distance traveled by the customer. We observe in the lower left corner that, when both OOH density and discounts are low, customers are less inclined to travel to OOH locations as indicated by the lighter color.

Moving to the right from the left lower corner, we observe the color shifts from light to a darker color. This indicates that discounts can convince modeled customers to accept a more remote OOH location.

When moving up from the lower left corner, we observe that the color also gets darker without providing higher discounts. This indicates that a higher OOH density

![](images/954ef13a020f451e09f9034cbac615a94436364b984ad096dd1671e3736ec163.jpg)  
Figure C.2. Analysis of the Distance Traveled to OOH Locations by Customers Given OOH Density and OOH Discount, Results Based on the RC Instances

requires lower discounts to reach the same OOH utilization compared with an area with lower OOH density. This aligns with the observations from Lyu and Teo (2022).

# Appendix D. DSPO Implementation Details

For the selection decision, $a ^ { s e l e c t i o n } .$ , we use the heuristic rule to offer the $N = 2 0$ closest OOH locations to the customer’s home address. For the synthetic case, this means that we always offer all OOH available locations as $| { \mathcal { L } } | < N$ .

For the pricing decision, $a ^ { p r i c i n g }$ , we collect a large data set of one million data points for the initial training phase. This data set is collected when using the NoPricing benchmark during simulation. DSPO uses a rectified linear unit (ReLU) activation function for the fully connected layers and no activation function for the output node as DSPO is a regression model. For the experiments with finitecapacity OOH locations, we add a small penalty to the cost function (see Equation (D.1)) during training. This penalty ensures that the model prefers to nudge customers to OOH locations that still have a lot of remaining capacity:

$$
P \left(k _ {t, l}\right) = \lambda_ {t} \frac {1}{1 + \exp \left(- w \left(\frac {k _ {t , l}}{C _ {l}} - \alpha\right)\right)}, \tag {D.1}
$$

where $P ( k _ { t , l } )$ is a penalty given the remaining capacity of location $l , C _ { l }$ is the maximum capacity of location $l , w$ is the steepness of the penalty increase, and $\alpha$ is a threshold parameter representing the fraction of the capacity at which the penalty starts to increase significantly. The penalty is weighted using $\lambda$ . After tuning, we found the following values: $w = \lceil 0 . 1 C _ { l } \rceil$ and $\alpha = 0 . 8$ , that is, the penalty function starts to increase noticeably when the locker is $8 0 \%$ full, and it increases sharply as it approaches full capacity. The weight $\lambda _ { t }$ starts at 0.1 and increases linearly by 0.001 at each time step. This way, the penalty has more weight at the end of the horizon. The model is trained using adaptive moment estimation (Adam) (Kingma and Ba 2014), and the goal is to minimize the Huber loss, which is less sensitive

to outliers compared with the $L _ { 1 }$ loss. The Huber loss function is defined by

$$
\mathcal {L} _ {\delta} (x) = \left\{ \begin{array}{l l} \frac {1}{2} x ^ {2} & \text {f o r} | x | \leq \delta , \\ \delta \left(| x | - \frac {1}{2} \delta\right) & \text {o t h e r w i s e}, \end{array} \right. \tag {D.2}
$$

where δ is a tunable parameter. The convergence of the Huber loss is depicted in Figure D.1 on the left for the synthetic case and on the right for the Amazon case. The figures show the result over five training seeds.

# Appendix E. Benchmarks

We describe four benchmarks: the Hindsight and Foresight benchmarks in Section E.1, the linear benchmark in Section E.2, and PPO in Section E.3. We provide a detailed description and show training and loss curves.

# E.1. Hindsight and Foresight Benchmark

The Hindsight and Foresight benchmarks are based on the policies proposed in Yang et al. (2016). In their paper, the estimated costs of inserting a customer into a time slot are calculated. Instead of calculating the costs over time slots, we calculate the costs per insertion of a new delivery location $k$ . The estimated costs of adding a delivery location $k$ are denoted by $\hat { C } _ { k , t } ^ { H i n d s i g h t }$ and $\hat { C } _ { k , t } ^ { F o r e s i g h \breve { t } } ,$ respectively. Obtaining $\hat { C } _ { k , t } ^ { H i n d s i g h t }$ is relatively straightforward: the additional travel time of inserting a customer into route $R _ { t - 1 }$ are first calculated, after which the salary, fuel, and service time costs related to delivery location $k$ can be determined. The preliminary route $R _ { t - 1 }$ contains a routing plan serving all known customers, potentially with multiple vehicles. The Foresight benchmark is similar but includes a weighted cost term:

$$
\hat {C} _ {k, t} ^ {\text {F o r e s i g h t}} = \left(1 - \theta_ {t} ^ {H}\right) \hat {C} _ {k, t} ^ {\text {H i n d s i g h t}} + \theta_ {t} ^ {H} \frac {1}{| \mathcal {P} |} \sum_ {p \in \mathcal {P}} f (k, p), \tag {E.1}
$$

where $f ( k , p )$ represents the insertion costs of feasibly inserting delivery location $k$ into historic route $p$ . We use a pool

![](images/87a8aa9ca26ff61b1e646861e82e0231a32ea18100a355f79dfda6cabc76d519.jpg)  
Figure D.1. Convergence Curves of the Huber Training Loss of DSPO for the Synthetic and Amazon Cases Reported over Five Training Seeds with a Training Seed Shaded Area of Two Standard Deviations

![](images/e387c2981612487344c123f8e10dbaf2149df19b6421c80cfc87b1fd10c7aea1.jpg)

![](images/98ec161ff5db36e4afb2d6d6c3a6aaac077031437fe3cba7c12abff5306df367.jpg)  
Figure E.1. Convergence Curves of the Huber Training Loss of the Linear Benchmark for the Synthetic and Amazon Cases Reported over Five Training Seeds with a Training Seed Shaded Area of Two Standard Deviations

![](images/a8f4331e2e217c67e201c2afa56f6a7054215b8375733d7d57e0329d76b641f2.jpg)

of 10 historic routes, $R _ { T } \in \mathcal { R } _ { T } ,$ obtained using the StaticPricing benchmark on the training data set. Note that these are final routing plans as denoted by the capital T. The weight $\theta _ { t } ^ { H }$ is initialized on the value $\theta _ { 0 } ^ { \bar { H } }$ and decreased after every new insertion by $\Delta \theta ^ { H }$ . Both parameters are considered tunable hyperparameters; see Appendix F for the tuning results. We preselect such that only delivery options that still have remaining capacity are considered. For further details on these policies in a time-slotting setting, we refer to Yang et al. (2016).

# E.2. Linear Benchmark

For the linear benchmark, we flatten the encoded state $\phi ( s _ { t } )$ as depicted in Figure 3. This flattened state is directly fed to the linear regression model, which yields the cost approximation $\hat { C } _ { k , t } ^ { L i n \overline { { e a r } } }$ per delivery option k. We use stochastic gradient descent to train the model and minimize the Huber loss function. Figure E.1 shows the convergence curves of the loss for the synthetic case on the left and the Amazon case on the right.

# E.3. PPO Benchmark

For PPO, we provide the following state information: (i) the coordinates of the home address of the new customer, (ii) the coordinates of the 20 delivery stops in $B _ { t - 1 }$ closest to the customer’s home address, and (iii) the remaining capacity of the $N$ closest OOH locations. The continuous feature values are represented using the $f ^ { \mathrm { t h } }$ order coupled Fourier basis, which is a linear approximation using the terms of the Fourier series as features; see Konidaris et al. (2011). We consider $f ,$ the order of the Fourier basis, to be a tunable hyperparameter; see Appendix F for the tuning results. Both the actor and critic use a neural network with three hidden layers and ReLU activation. The actor output layer uses a sigmoid activation function, which ensures that the Gaussian mean decision is in the range [0, 1]. As a final step, we multiply the pricing vector with a scaling factor, corresponding to the pricing bounds [�10, 2]. The price for home delivery is multiplied by two, and the prices for OOH delivery locations are multiplied by $- 1 0$ . We provide the same penalty function for capacitated OOH locations to

![](images/2ef43499d8974bdb04274496ca118f0cbe79a7f444cd84d652354e67f4582ecc.jpg)  
Figure E.2. Convergence Curves of PPO for the Synthetic and Amazon Cases Reported over Five Training Seeds with a Training Seed Shaded Area of Two Standard Deviations

![](images/1c17e536d78288b174ac3841ceeb675f97ad31a7efb162ef31aecfd352862250.jpg)

PPO as we do for DSPO; see Appendix D. We train the critic and actor using Adam (Kingma and Ba 2014) and calculate critic loss with the Huber loss function.

Algorithm E.1 outlines the PPO algorithm as detailed in Schulman et al. (2017). We begin by initializing the network weights w for the critic and u for the actor (step 1), followed by setting hyperparameters such as the Gaussian standard deviation $\sigma$ and the learning rates $\alpha _ { \mathrm { c r } }$ and $\alpha _ { \mathrm { a c } }$ for the critic and actor, respectively (step 2). After initializing a state $s _ { 0 }$ (step 3), we enter a loop for each time step in the booking horizon (step 4). A continuous decision a is generated by sampling from the policy $\pi _ { \mathbf { \theta } } ,$ guided by the learned mean $\mu$ and standard deviation $\sigma _ { \varepsilon }$ along with the weights $\theta$ (step 5). Upon applying decision a to the environment, we observe the transition to the subsequent state $s _ { t + 1 }$ (step 6), which we then store in the trajectory buffer $\tau$ (step 7). Every $T$ steps, updates are made to the actor and critic networks (step 8). Advantages are computed using the truncated generalized advantage estimation (GAE) as follows (step 9):

$$
\hat {A} _ {t} \left(r, s _ {t}, a, s _ {t + 1}\right) = \sum_ {t ^ {\prime} = t} ^ {T} (\lambda \gamma) ^ {t ^ {\prime} - t} \delta_ {t ^ {\prime}}, \tag {E.2}
$$

where $\lambda$ is the temporal difference discount parameter and δ is defined by

$$
\delta = r + \gamma Q \left(s _ {t + 1}, a, \boldsymbol {w}\right) - Q \left(s _ {t}, a, \boldsymbol {w}\right). \tag {E.3}
$$

We then proceed to optimize the policy loss (step 10) as described in Schulman et al. (2017) and the value loss (step 11) based on the $n$ -step return:

$$
G _ {t} ^ {n} = r _ {t} + \gamma r _ {t + 1} + \dots + \gamma^ {n} Q _ {w} \left(s _ {t + n}, a\right). \tag {E.4}
$$

The trajectory buffer $\tau$ is emptied thereafter (step 12).

# Algorithm E.1 (PPO Algorithm (Gaussian Policy))

1: Initialize critic and actor network weights w, u   
2: Set hyperparameters: $\sigma , \alpha _ { \mathrm { c r } } , \alpha _ { \mathrm { a c } }$   
3: for each episode do   
4: for $\mathrm { t } = 1 , 2 ,$ , … , T do   
5: $a \gets \pi _ { \pmb { \theta } } ( s _ { t } )$ (based on s)   
6: Apply a to environment, observe successor state $s _ { t + 1 }$   
7: Store $( s _ { t } , a , s _ { t + 1 } ) _ { t }$ in $\tau$   
8: Compute negative rewards $r$ based on Equation (4) and add them to $\mathcal { T } _ { T }$   
9: Compute advantages $\hat { A } _ { t } ( r , s _ { t } , a , s _ { t + 1 } )$ and $\log ( \pi _ { \pmb { \theta } } ( a ) )$   
10: Optimize clipped policy loss based on $\hat { A } ( \boldsymbol r , \boldsymbol s _ { t } ,$ $a , s _ { t + 1 } )$ ) and $\log ( \pi _ { \pmb { \theta } } ( a ) )$   
11: Optimize critic loss based on $n$ -step return   
12: Empty T

The convergence of the costs over training episodes is depicted in Figure E.2 on the left for the synthetic case and on the right for the Amazon case. PPO requires many episodes to train, probably caused by the sparse cost structure of the problem. For the synthetic case, PPO seems to require 200, 000 episodes to explore, after which it suddenly jumps to a well-performing policy. For the Amazon case, PPO is unable to converge to a performant policy within 600,000 episodes.

# Appendix F. Hyperparameters

In this section, we provide an overview of the hyperparameter tuning results for the Foresight benchmark, linear benchmark, PPO, and DSPO; see Table F.1.

The Foresight benchmark has two parameters: the initial weight given to the insertion costs in the current route

Table F.1. Hyperparameter Tuning   

<table><tr><td rowspan="2"></td><td rowspan="2">Hyperparameters</td><td rowspan="2">Set of values</td><td colspan="2">Selected values</td></tr><tr><td>Synthetic case</td><td>Amazon case</td></tr><tr><td rowspan="2">FB</td><td>θ0H(initial insertion weight)</td><td>{1.0,0.75,0.25}</td><td>1.0</td><td>1.0</td></tr><tr><td>ΔθH insertion weight update)</td><td>{θ0H/E[D],0.05,0.1}</td><td>θ0H/E[D]</td><td>θ0H/E[D]</td></tr><tr><td rowspan="2">LB</td><td>Huber loss δ</td><td>{0.5,0.75,1.0,1.35,1.5}</td><td>1.0</td><td>1.0</td></tr><tr><td>αLB (learning rate)</td><td>{10-2,10-3,10-4}</td><td>10-2</td><td>10-3</td></tr><tr><td rowspan="11">PPO</td><td>γ (discount factor)</td><td>{0.9,0.99}</td><td>0.99</td><td>0.99</td></tr><tr><td>αPPO (learning rate critic)</td><td>{10-2,10-3,10-4,10-5}</td><td>10-4</td><td>10-4</td></tr><tr><td>αPPO (learning rate actor)</td><td>{10-2,10-3,10-4,10-5}</td><td>10-5</td><td>10-5</td></tr><tr><td>σ</td><td>{†,0.25,0.5,1}</td><td>0.25</td><td>0.5</td></tr><tr><td>Huber loss δ</td><td>{0.5,0.75,1.0,1.35,1.5}</td><td>1.0</td><td>1.0</td></tr><tr><td>Number of actor NN nodes/layer</td><td>{8,16,32}</td><td>8</td><td>16</td></tr><tr><td>Number of critic NN nodes/layer</td><td>{8,16,32,64}</td><td>16</td><td>32</td></tr><tr><td>Batch size</td><td>{32,64,128}</td><td>128</td><td>128</td></tr><tr><td>f (Fourier order)</td><td>{2,3,4}</td><td>3</td><td>3</td></tr><tr><td>Clipping factor</td><td>{0.1,0.2,0.3}</td><td>0.2</td><td>0.2</td></tr><tr><td>GAE λ</td><td>{0.9,0.95,0.99,1.0}</td><td>0.95</td><td>0.95</td></tr><tr><td rowspan="7">DSPO</td><td>M (number of grids state spatial dimension)</td><td>{25,100,900,1600,3600}</td><td>100</td><td>1,600</td></tr><tr><td>DT (number of layers state temporal dimension)</td><td>{1,2,3,4,5,6,7,9,10}</td><td>3</td><td>8</td></tr><tr><td>Batch size</td><td>{32,64,128}</td><td>64</td><td>128</td></tr><tr><td>αDSPO (learning rate)</td><td>{10-2,10-3,10-4,10-5}</td><td>10-3</td><td>10-3</td></tr><tr><td>Huber loss δ</td><td>{0.5,0.75,1.0,1.35,1.5}</td><td>1.0</td><td>1.0</td></tr><tr><td>Number of NN nodes/FC layer</td><td>{32,64,128,256}</td><td>128</td><td>256</td></tr><tr><td>Dropout rate FC layer</td><td>{0,0.01,0.05,0.1}</td><td>0.0</td><td>0.05</td></tr></table>

Note. FB, foresight benchmark; LB, linear benchmark.

$R _ { t - 1 } , \theta _ { 0 } ^ { H } ,$ , and by how much this weight is decreased after every new customer insertion, $\Delta \theta ^ { H }$ . We found that setting $\theta _ { 0 } ^ { H } = 1 . 0$ and $\Delta \theta ^ { H } = \frac { \theta _ { 0 } ^ { H } } { \mathbb { E } [ D ] }$ = � θH0E[D] yields the best results, which is the same setting as in Yang et al. (2016).

For PPO, we study both (i) learning the second moment of the Gaussian distribution, $\sigma ,$ and (ii) setting $\sigma$ to a constant value. In Table F.1, a $\dagger$ indicates that $\sigma$ was learned by the actor. We maintain a learning rate for the actor that is 10 times smaller than that of the critic, to guarantee that the critic’s provided values are up to date. Both the actor and critic use a fully connected neural network with three hidden layers and ReLU activation.

Apart from the DSPO hyperparameters in Table F.1, we use the following default settings for the CNN. The first convolutional layer has 32 filters for the synthetic case and 64 filters for the Amazon case. The second convolutional layer has two times more filters compared with the first layer. The kernel has dimension (3, 3) for the convolution layers and (2, 2) for the pooling layer. The stride of the kernel is (1, 1), padding is (1, 1), and we do not use dilation.

# Appendix G. Complementary Figures

In this section, we provide complementary figures. Figure G.1 provides a heat map of the number of delivery stops per cell over the complete service area of Seattle on an average day. Note that the figure is tilted by $9 0 ^ { \circ }$ such that the north is oriented to the right.

Figures G.2 and G.3 depict heat maps of the costs predicted by DSPO at cutoff time and the OOH utilization when using DSPO, respectively. These exemplary figures help to understand the decisions resulting from DSPO.

For visualization purposes, we zoom in on an exemplary part of Seattle: the densely populated neighborhoods in northern Seattle and the suburb of east Seattle, Kirkland. The suburb has fewer customers and is harder to reach because it is separated from Seattle by Lake Washington.

Figure G.2 shows that DSPO predicts higher delivery costs for more remote areas and areas that have lower OOH density. The Kirkland areas and the parts with lower OOH density have higher predicted costs. DSPO provides higher discounts for choosing OOH delivery to customers in these areas.

Figure G.3 shows that, in areas with high OOH density, the utilization of such delivery options is highest, ranging from $6 0 \%$ to $8 0 \%$ . Conversely, in areas with sparser OOH locations, the usage rate is lower. Interestingly, the utilization in remote Kirkland areas is slightly higher compared with the low OOH density areas in northern Seattle. Probably, customers in Kirkland areas are marked as more expensive to serve because they are more remote, and consequently, higher incentives are given to those customers.

Figure G.4 shows a boxplot of the online step time in milliseconds for all dynamic pricing methods. The online step time is the time required to obtain a single decision. This is relevant in light of website load times required for providing the delivery options to the online shoppers. Although the route insertion calculation is, in essence, relatively cheap in terms of computational effort, it does require the calculation of new travel times; that is, in a practical situation it requires an API request. The linear benchmark, PPO, and DSPO do not require this calculation and, hence, provide faster results. We note that, in absolute terms, the differences between the policies are relatively small.

![](images/5cb1ec3e4cedc13aff898bc438efab8e734d88b2043a5fdec6debb4b12857afd.jpg)  
Figure G.1. (Color online) Seattle Heat Map of Delivery Stops on an Average Day

Figure G.2. (Color online) Northern Seattle Heat Map of the Estimated Delivery Costs According to DSPO at the Cutoff Time   
![](images/040e0349576c38310ea4b07105d8fcc9d4d19fe9c1b6098483d4d380d29170eb.jpg)  
OOHdelivery location

Figure G.3. (Color online) Northern Seattle Heat Map of the Utilization of OOH Delivery When Using DSPO   
![](images/3bf03b5c8a4e038d8af1f231ff16688bb1c5f8c318cc808b0420685133420b29.jpg)  
OOH delivery location

![](images/d404d2afaf4960c48e7942f6bdef321ddf624d2bc37dc1e77a3703cbeb219486.jpg)  
Figure G.4. Step Time in Milliseconds of the Studied Dynamic Pricing Methods (Amazon Case)

# References

Agatz N, Campbell A, Fleischmann M, Savelsbergh M (2011) Time slot management in attended home delivery. Transportation Sci. 45(3):435–449.   
Akamai (2017) Akamai online retail performance report: Milliseconds are critical. Accessed October 7, 2024, https://www.ir. akamai.com/news-releases/news-release-details/akamai-onlineretail-performance-report-milliseconds-are.   
Allied Market Research (2022) CEP market opportunities and forecasts. Accessed October 7, 2024, https://www.alliedmarketresearch. com/courier-express-and-parcel-market-A11516.   
Amazon (2024a) Amazon delivery points. Accessed October 7, 2024, https://www.amazon.com/ulp.   
Amazon (2024b) Amazon hub. Accessed October 7, 2024, https:// www.amazon.com/b?ie=UTF8\&node=13853235011.   
Arnold F, Cardenas I, So¨rensen K, Dewulf W (2018) Simulation of B2C e-commerce distribution in Antwerp using cargo bikes and delivery points. Eur. Transport Res. Rev. 10:2.   
Asdemir K, Jacob VS, Krishnan R (2009) Dynamic pricing of multiple home delivery options. Eur. J. Oper. Res. 196(1):246–257.   
Ausseil R, Pazour JA, Ulmer MW (2022) Supplier menus for dynamic matching in peer-to-peer transportation platforms. Transportation Sci. 56(5):1304–1326.   
Campbell A, Savelsbergh M (2006) Incentive schemes for attended home delivery services. Transportation Sci. 40(3):327–341.   
Chen Q, Conway A, Cheng J (2017) Parking for residential delivery in New York City: Regulations and behavior. Transport Policy 54:53–60.   
Corless RM, Gonnet GH, Hare DEG, Jeffrey DJ, Knuth DE (1996) On the LambertW function. Adv. Comput. Math. 5(1):329–359.   
Dalla Chiara G, Goodchild A (2020) Do commercial vehicles cruise for parking? Empirical evidence from Seattle. Transport Policy 97:26–36.   
Dalla Chiara G, Krutein KF, Ranjbari A, Goodchild A (2021) Understanding urban commercial vehicle driver behaviors and decision making. Transportation Res. Rec. 2675(9):608–619.   
Deutsch Y, Golany B (2018) A parcel locker network as a solution to the logistics last mile problem. Internat. J. Production Res. 56(1–2):251–261.   
Dong L, Kouvelis P, Tian Z (2009) Dynamic pricing and inventory control of substitute products. Manufacturing Service Oper. Management 11(2):317–339.   
Dumez D, Lehue´de ´ F, Pe´ton O (2021) A large neighborhood search approach to the vehicle routing problem with delivery options. Transportation Res. Part B Methodological 144:103–132.   
Edwards J, Mckinnon A, Cherrett T, Mcleod F, Song L (2009) The impact of failed home deliveries on carbon emissions: Are collection/delivery points environmentally-friendly alternatives? Logist. Res. Network Annual Conf. (Cardiff, UK).   
Enthoven DLJU, Jargalsaikhan B, Roodbergen KJ, uit het Broek MAJ, Schrotenboer AH (2020) The two-echelon vehicle routing problem with covering options: City logistics with cargo bikes and parcel lockers. Comput. Oper. Res. 118:104919.   
Galiullina A, Mutlu N, Kinable J, Van Woensel T (2024) Demand steering in a last-mile delivery problem with home and pickup point delivery options. Transportation Sci. 58(2):454–473.   
Gehring H, Homberger J (2002) Parallelization of a two-phase metaheuristic for routing problems with time windows. J. Heuristics 8(3):251–276.   
Ghaderi H, Zhang L, Tsai PW, Woo J (2022) Crowdsourced lastmile delivery with parcel lockers. Internat. J. Production Econom. 251:108549.   
Goodfellow I, Bengio Y, Courville A (2016) Deep Learning (MIT Press, Cambridge, MA).   
Grabenschweiger J, Doerner KF, Hartl RF, Savelsbergh MWP (2021) The vehicle routing problem with heterogeneous locker boxes. Central Eur. J. Oper. Res. 29(1):113–142.

Janinhoff $\scriptstyle \mathrm { { L , } }$ Klein R (2023) Stochastic location routing for out-ofhome delivery networks. Preprint, submitted December 20, https://dx.doi.org/10.2139/ssrn.4654115.   
Janinhoff ${ \mathrm { L } } ,$ Klein R, Sailer D, Schoppa JM (2024) Out-of-home delivery in last-mile logistics: A review. Comput. Oper. Res. 168:106686.   
Jiang L, Dhiaf M, Dong J, Liang C, Zhao S (2020) A traveling salesman problem with time windows for the last mile delivery in online shopping. Internat. J. Production Res. 58(16):5077–5088.   
Kahr M (2022) Determining locations and layouts for parcel lockers to support supply chain viability at the last mile. Omega 113:102721.   
Karabulut E, Gholizadeh F, Akhavan-Tabatabaei R (2022) The value of adaptive menu sizes in peer-to-peer platforms. Transportation Res. Part C Emerging Tech. 145:103948.   
Kedia A, Kusumastuti D, Nicholson A (2017) Acceptability of collection and delivery points from consumers’ perspective: A qualitative case study of Christchurch City. Case Studies Transport Policy 5(4):587–595.   
Kingma D, Ba J (2014) Adam: A method for stochastic optimization. Preprint, submitted December 22, https://arxiv.org/abs/1412.6980.   
Klein R, Mackert J, Neugebauer M, Steinhardt C (2018) A modelbased approximation of opportunity cost for dynamic pricing in attended home delivery. OR Spectrum 40(4):969–996.   
Klein R, Neugebauer M, Ratkovitch D, Steinhardt C (2019) Differentiated time slot pricing under routing considerations in attended home delivery. Transportation Sci. 53(1):236–255.   
Koch S, Klein R (2020) Route-based approximate dynamic programming for dynamic pricing in attended home delivery. Eur. J. Oper. Res. 287(2):633–652.   
Konidaris G, Osentoski S, Thomas P (2011) Value function approximation in reinforcement learning using the Fourier basis. Proc. Conf. AAAI Artificial Intelligence, vol. 25 (AAAI Press, Palo Alto, CA), 380–385.   
Last Mile Experts (2022) Out of home delivery in Europe. Accessed October 7, 2024, https://lastmileexperts.com/reports-case-studies/.   
Li Z, Liu F, Yang W, Peng S, Zhou J (2022) A survey of convolutional neural networks: Analysis, applications, and prospects. IEEE Trans. Neural Networks Learn. Systems 33(12): 6999–7019.   
Lin Y, Wang Y, Lee LH, Chew EP (2022) Profit-maximizing parcel locker location problem under threshold luce model. Transportation Res. Part E Logist. Transportation Rev. 157:102541.   
Liu Y, Ye Q, Escribano-Macias J, Feng Y, Candela E, Angeloudis P (2023) Route planning for last-mile deliveries using mobile parcel lockers: A hybrid q-learning network approach. Transportation Res. Part E Logist. Transportation Rev. 177:103234.   
Loquate (2021) Fixing failed deliveries. Accessed October 7, 2024, https://www.loqate.com/resources/ebooks-and-reports/fixingfailed-deliveries/.   
Luo R, Ji S, Ji Y (2022) An active-learning Pareto evolutionary algorithm for parcel locker network design considering accessibility of customers. Comput. Oper. Res. 141:105677.   
Lyu G, Teo CP (2022) Last mile innovation: The case of the locker alliance network. Manufacturing Service Oper. Management 24(5): 2425–2443.   
Mancini S, Gansterer M (2021) Vehicle routing with private and shared delivery locations. Comput. Oper. Res. 133:105361.   
Mancini S, Gansterer M, Triki C (2023) Locker box location planning under uncertainty in demand and capacity availability. Omega 120:102910.   
Merchan D, Arora J, Pachon J, Konduri K, Winkenbach M, Parks S, Noszek J (2022) 2021 Amazon last mile routing research challenge: Data set. Transportation Sci. 58(1):8–11.   
Mnih V, Kavukcuoglu K, Silver D, Graves A, Antonoglou I, Wierstra D, Riedmiller M (2013) Playing Atari with deep reinforcement learning. Preprint, submitted December 19, https://arxiv. org/abs/1312.5602.

Molga M, Smutnicki C (2005) Test functions for optimization needs. Accessed October 7, 2024, https://www.sfu.ca/~ssurjano/camel6. html.   
Pan S, Zhang L, Thompson RG, Ghaderi H (2021) A parcel network flow approach for joint delivery networks using parcel lockers. Internat. J. Production Res. 59(7):2090–2115.   
Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, et al. (2019) Pytorch: An imperative style, high-performance deep learning library. Wallach H, Larochelle H, Beygelzimer A, d’Alche´- Buc F, Fox E, Garnett R, eds. Adv. Neural Inform. Processing Systems, vol. 32 (Curran Associates, Inc., Red Hook, NY), 8024–8035.   
Peng X, Zhang L, Thompson RG, Wang K (2023) A three-phase heuristic for last-mile delivery with spatial-temporal consolidation and delivery options. Internat. J. Production Econom. 266:109044.   
Ranjbari A, Diehl C, Dalla Chiara G, Goodchild A (2023) Do parcel lockers reduce delivery times? Evidence from the field. Transportation Res. Part E Logist. Transportation Rev. 172:103070.   
Raviv T (2023) The service points’ location and capacity problem. Transportation Res. Part E Logist. Transportation Rev. 176:103216.   
Redlands ESRI (2024) Arcgis desktop: Release 10.8.   
Savelsbergh M, Van Woensel T (2016) 50th anniversary invited article—City logistics: Challenges and opportunities. Transportation Sci. 50(2):579–590.   
Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O (2017) Proximal policy optimization algorithms. Preprint, submitted July 20, https://arxiv.org/abs/1707.06347.   
Schwerdfeger S, Boysen N (2022) Who moves the locker? A benchmark study of alternative mobile parcel locker concepts. Transportation Res. Part C Emerging Tech. 142:103780.   
Sethuraman S, Bansal A, Mardan S, Resende MGC, Jacobs TL (2024) Amazon locker capacity management. INFORMS J. Appl. Anal. Forthcoming.   
Silver D, Lever G, Heess N, Degris T, Wierstra D, Riedmiller M (2014) Deterministic policy gradient algorithms. Xing EP, Jebara T, eds. Proc. 31st Internat. Conf. Machine Learn., vol. 32 (PLMR, New York), 387–395.   
Sitek P, Wikarek J (2019) Capacitated vehicle routing problem with pick-up and alternative delivery (CVRPPAD): Model and implementation using hybrid approach. Ann. Oper. Res. 273(1):257–277.   
Song L, Cherrett T, McLeod F, Guan W (2009) Addressing the last mile problem: Transport impacts of collection and delivery points. Transportation Res. Rec. 2097(1):9–18.

Strauss A, Gu¨ lpinar N, Zheng Y (2021) Dynamic pricing of flexible time slots for attended home delivery. Eur. J. Oper. Res. 294(3): 1022–1041.   
Train KE (2009) Discrete Choice Methods with Simulation, 2nd ed. (Cambridge Books, Cambridge University Press, New York).   
Ulmer MW (2020) Dynamic pricing and routing for same-day delivery. Transportation Sci. 54(4):1016–1033.   
Ulmer MW, Streng S (2019) Same-day delivery with pickup stations and autonomous vehicles. Comput. Oper. Res. 108:1–19.   
Vidal T (2022) Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. Comput. Oper. Res. 140:105643.   
Vidal T, Crainic TG, Gendreau M, Lahrichi N, Rei W (2012) A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. Oper. Res. 60(3):611–624.   
Vinsensius A, Wang Y, Chew EP, Lee LH (2020) Dynamic incentive mechanism for delivery slot management in e-commerce attended home delivery. Transportation Sci. 54(3):567–587.   
Vukicevi ´ c ´ M, Ratli M, Rivenq A, Zrikem M (2023) Covering delivery problem with electric vehicle and parcel lockers: Variable neighborhood search approach. Comput. Oper. Res. 157:106263.   
Xu X, Shen Y, Chen W(A), Gong Y, Wang H, (2021) Data-driven decision and analytics of collection and delivery point location problems for online retailers. Omega 100:102280.   
Yang X, Strauss AK (2017) An approximate dynamic programming approach to attended home delivery management. Eur. J. Oper. Res. 263(3):935–945.   
Yang X, Strauss AK, Currie CSM, Eglese R (2016) Choice-based demand management and vehicle routing in e-fulfillment. Transportation Sci. 50(2):473–488.   
Yildiz B, Savelsbergh M (2020) Pricing for delivery time flexibility. Transportation Res. Part B Methodological 133:230–256.   
Zhang W, Xu M, Wang S (2023) Joint location and pricing optimization of self-service in urban logistics considering customers’ choice behavior. Transportation Res. Part E Logist. Transportation Rev. 174:103128.   
Zhou L, Baldacci R, Vigo D, Wang X (2018) A multi-depot twoechelon vehicle routing problem with delivery options arising in the last mile distribution. Eur. J. Oper. Res. 265(2): 765–778.