## Reinforcement Learning


* [MIT 6.S191 Lecture 6: Deep Reinforcement Learning ](https://www.youtube.com/watch?v=xWe58WGWmlk&index=4&list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs)
  * RL: given data, choose action to maximize expected long-term reward
  * quantize interactions into episodes: s0,a0,r0, s1,a1,r1, ..., sT,aT,rT
  * key to RL: transition function *p(st+1,rt|st,at)*
  * objective in RL: finding policy *\( \pi(s)=p(a|s) \)* to maximize reward
  * two major methods to find policy
    * policy learning: find *pi(s)* directly
    * value learning: given state and action estimate max future reward
  * *Q(st,at)* value: expected future reward at state *st* taking action *at* assuming subsequenct actions are going to be perfect
  
    
  


