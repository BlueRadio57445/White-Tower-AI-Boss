import numpy as np
import random
import matplotlib.pyplot as plt
import os, time
from tqdm import tqdm

# --- 1. 數值工具 ---
def squared_prob(logits):
    """
    硬體友善版本：使用平方機率分布（無死亡神經元問題）
    s_i = logit_i^2 + epsilon
    P_i = s_i / sum(s_k)
    """
    scores = logits**2 + 1e-5  # 平方 + epsilon
    total_score = np.sum(scores)
    probs = scores / total_score
    return probs, scores, total_score, logits  # 返回 logits 用於梯度計算

def gaussian_log_prob(x, mu, sigma):
    """計算高斯分佈的對數機率密度"""
    var = sigma**2
    log_prob = -((x - mu)**2) / (2 * var) - np.log(sigma * np.sqrt(2 * np.pi))
    return log_prob

# --- 2. 混合動作空間的線性 PPO Agent (平方機率版) ---
class HybridLinearPPO_Squared:
    def __init__(self, n_feat, n_discrete_acts, gamma=0.99, lmbda=0.95, eps=0.2):
        """
        混合動作空間 PPO (平方機率版本):
        - 離散動作: 使用平方機率分布（解決死亡神經元問題）
        - 連續動作: 施法時的瞄準偏移
        """
        self.n_discrete_acts = n_discrete_acts
        
        # 離散 Actor 權重
        self.w_actor_discrete = np.random.randn(n_discrete_acts, n_feat) * 0.01
        
        # 連續 Actor 權重
        self.w_actor_continuous_mu = np.random.randn(n_feat) * 0.01
        
        # 連續動作的標準差
        self.sigma = 0.6
        self.sigma_min = 0.15
        self.sigma_decay = 0.9998
        
        # Critic 權重
        self.w_critic = np.random.randn(n_feat) * 0.01
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.buffer = []

    def get_action(self, obs):
        """
        輸出混合動作:
        - 使用平方機率分布（只用加減乘除，無死亡神經元）
        """
        # 1. 離散動作決策（平方機率）
        logits = np.dot(self.w_actor_discrete, obs)
        probs_discrete, scores, sum_scores, logits_saved = squared_prob(logits)
        
        a_discrete = np.random.choice(len(probs_discrete), p=probs_discrete)
        
        # 2. 連續動作決策
        mu = np.dot(self.w_actor_continuous_mu, obs)
        a_continuous = np.random.normal(mu, self.sigma)
        
        # 3. 價值估計
        v = np.dot(self.w_critic, obs)
        
        # 保存 logits 用於梯度計算
        return a_discrete, a_continuous, probs_discrete[a_discrete], mu, v, logits_saved

    def store(self, transition):
        self.buffer.append(transition)

    def update(self, lr_a_discrete=0.003, lr_a_continuous=0.002, lr_c=0.007):
        if not self.buffer: 
            return
            
        # Buffer 格式: (obs, a_d, a_c, prob_d, mu, v, logits, r)
        states, a_discretes, a_continuouses, old_probs_discrete, old_mus, values, old_logits_list, rewards = zip(*self.buffer)
        
        # --- 計算 GAE 優勢函數 ---
        advantages = []
        gae = 0
        next_v = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_v - v
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            next_v = v
        
        advs = np.array(advantages)
        returns = advs + np.array(values)

        # --- 更新權重 ---
        for i in range(len(self.buffer)):
            s = states[i]
            a_d = a_discretes[i]
            a_c = a_continuouses[i]
            target_v = returns[i]
            op_d = old_probs_discrete[i]
            o_mu = old_mus[i]
            adv = advs[i]
            
            # 1. Critic MSE 更新
            critic_grad = (target_v - np.dot(self.w_critic, s)) * s
            critic_grad = np.clip(critic_grad, -5.0, 5.0)
            self.w_critic += lr_c * critic_grad
            
            # 2. 離散 Actor 更新（平方機率梯度）
            logits_new = np.dot(self.w_actor_discrete, s)
            probs_new, scores_new, sum_new, _ = squared_prob(logits_new)
            
            ratio_discrete = probs_new[a_d] / (op_d + 1e-8)
            
            clipped_ratio_d = np.clip(ratio_discrete, 1-self.eps, 1+self.eps)
            if (ratio_discrete * adv <= clipped_ratio_d * adv) or (1-self.eps < ratio_discrete < 1+self.eps):
                # ✨ 平方機率的梯度公式
                # Grad_a = (2 * logit_a / S) * (1/P_a - 1) * x
                inv_sum = 1.0 / (sum_new + 1e-8)
                inv_prob = np.clip(1.0 / (probs_new[a_d] + 1e-8), 0, 50)
                
                # 更新選中的動作
                grad_selected = adv * (2 * logits_new[a_d] * inv_sum) * (inv_prob - 1) * s
                self.w_actor_discrete[a_d] += lr_a_discrete * grad_selected
                
                # 更新其他動作
                # Grad_j = -(2 * logit_j / S) * x
                for j in range(self.n_discrete_acts):
                    if j != a_d:
                        grad_other = -adv * (2 * logits_new[j] * inv_sum) * s
                        self.w_actor_discrete[j] += lr_a_discrete * grad_other
            
            # 3. 連續 Actor PPO Clip 更新
            new_mu = np.dot(self.w_actor_continuous_mu, s)
            
            old_log_prob = gaussian_log_prob(a_c, o_mu, self.sigma)
            new_log_prob = gaussian_log_prob(a_c, new_mu, self.sigma)
            ratio_continuous = np.exp(new_log_prob - old_log_prob)
            
            clipped_ratio_c = np.clip(ratio_continuous, 1-self.eps, 1+self.eps)
            if (ratio_continuous * adv <= clipped_ratio_c * adv) or (1-self.eps < ratio_continuous < 1+self.eps):
                grad = (a_c - new_mu) / (self.sigma**2) * s
                self.w_actor_continuous_mu += lr_a_continuous * adv * grad
                
        # 探索率衰減
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
                
        self.buffer = []

# --- 3. 2D RPG 環境 (支援混合動作空間) ---
class RPG2DEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_angle = 0.0
        self.monster_pos = np.array([8.0, 8.0])
        self.blood_pos = np.array([2.0, 8.0])
        self.wind_up = 0
        return self._get_obs()

    def _get_obs(self):
        rel_m = (self.monster_pos - self.agent_pos) / self.size
        rel_b = (self.blood_pos - self.agent_pos) / self.size
        
        # 基礎特徵：相對位置
        monster_dx, monster_dy = rel_m[0], rel_m[1]
        blood_dx, blood_dy = rel_b[0], rel_b[1]
        
        # 距離特徵
        dist_to_monster = np.linalg.norm(self.monster_pos - self.agent_pos) / self.size
        dist_to_blood = np.linalg.norm(self.blood_pos - self.agent_pos) / self.size
        
        # 角度特徵
        angle_to_monster = np.arctan2(self.monster_pos[1] - self.agent_pos[1], 
                                       self.monster_pos[0] - self.agent_pos[0])
        angle_to_blood = np.arctan2(self.blood_pos[1] - self.agent_pos[1], 
                                     self.blood_pos[0] - self.agent_pos[0])
        
        # 相對角度差
        relative_angle_monster = np.arctan2(np.sin(angle_to_monster - self.agent_angle),
                                           np.cos(angle_to_monster - self.agent_angle))
        relative_angle_blood = np.arctan2(np.sin(angle_to_blood - self.agent_angle),
                                         np.cos(angle_to_blood - self.agent_angle))
        
        # 是否在瞄準範圍內
        monster_in_sight = 1.0 if abs(relative_angle_monster) < 0.5 and dist_to_monster < 0.6 else 0.0
        
        # 是否接近邊界
        dist_to_wall = min(self.agent_pos[0], self.agent_pos[1], 
                          self.size - self.agent_pos[0], self.size - self.agent_pos[1]) / self.size
        
        # 玩家朝向編碼
        cos_angle = np.cos(self.agent_angle)
        sin_angle = np.sin(self.agent_angle)
        
        # 施法狀態
        casting_progress = self.wind_up / 4.0
        is_ready_to_cast = 1.0 if self.wind_up == 0 else 0.0
        
        # 組合所有特徵
        return np.array([
            monster_dx, monster_dy, blood_dx, blood_dy,
            dist_to_monster, dist_to_blood,
            relative_angle_monster, relative_angle_blood,
            cos_angle, sin_angle,
            monster_in_sight,
            dist_to_wall,
            casting_progress, is_ready_to_cast,
            1.0
        ])

    def step(self, a_discrete, a_continuous):
        """
        混合動作執行:
        - a_discrete: 0=前進, 1=左轉, 2=右轉, 3=施法
        - a_continuous: 連續值，只在施法時作為瞄準偏移
        """
        reward = -0.01
        event = ""
        
        # 0: 前進
        if a_discrete == 0:
            move = np.array([np.cos(self.agent_angle), np.sin(self.agent_angle)]) * 0.6
            old_pos = self.agent_pos.copy()
            self.agent_pos = np.clip(self.agent_pos + move, 0, self.size-1)
            
            # 檢查是否碰牆
            if not np.allclose(old_pos + move, self.agent_pos):
                reward -= 2.0
                event = "HIT WALL!"
            
        # 1: 左轉
        elif a_discrete == 1:
            self.agent_angle += 0.4
            
        # 2: 右轉
        elif a_discrete == 2:
            self.agent_angle -= 0.4
            
        # 3: 施法
        elif a_discrete == 3 and self.wind_up == 0:
            self.wind_up = 4
            aim_offset = np.clip(a_continuous, -0.5, 0.5)
            self.aim_angle = self.agent_angle + aim_offset
            event = "CASTING..."

        # 觸碰血包判定
        if np.linalg.norm(self.agent_pos - self.blood_pos) < 1.0:
            reward += 12
            event = "EAT BLOOD!"
            self.blood_pos = np.random.uniform(1, 9, 2)

        # 施法完成判定
        if self.wind_up > 0:
            self.wind_up -= 1
            if self.wind_up == 0:
                vec = self.monster_pos - self.agent_pos
                angle_to = np.arctan2(vec[1], vec[0])
                angle_diff = np.arctan2(np.sin(angle_to - self.aim_angle), 
                                       np.cos(angle_to - self.aim_angle))
                
                if abs(angle_diff) < 0.4 and np.linalg.norm(vec) < 6:
                    reward += 25
                    event = "KILLED MONSTER!"
                    self.monster_pos = np.random.uniform(1, 9, 2)
                else:
                    event = "MISSED..."

        return self._get_obs(), reward, event

# --- 4. 訓練與視覺化流程 ---
env = RPG2DEnv()
agent = HybridLinearPPO_Squared(n_feat=15, n_discrete_acts=4)
history = []

print("正在啟動 2D 訓練場... 平方機率 PPO 更新機制已就緒。")
print("離散動作: 前進/左轉/右轉/施法 (使用平方機率分布)")
print("連續動作: 施法時的瞄準偏移")
print("手工特徵: 15 維")
print("✨ 優勢: 無死亡神經元問題，只用加減乘除")
epochs = 12000

for ep in tqdm(range(epochs)):
    obs = env.reset()
    total_r = 0
    for _ in range(50):
        a_d, a_c, prob_d, mu, v, logits = agent.get_action(obs)
        next_obs, r, event = env.step(a_d, a_c)
        agent.store((obs, a_d, a_c, prob_d, mu, v, logits, r))
        obs = next_obs
        total_r += r
        
        # 渲染最後 10 輪
        if ep > epochs-10:
            os.system('cls' if os.name == 'nt' else 'clear')
            action_names = ["MOVE", "LEFT", "RIGHT", "CAST"]
            print(f"Ep: {ep} | Reward: {total_r:.1f} | Sigma: {agent.sigma:.3f}")
            print(f"Action: {action_names[a_d]} | Continuous: {a_c:.2f} | Event: {event}")
            
            # 創建帶牆壁的 canvas
            canvas = np.full((10, 10), ".")
            canvas[int(env.monster_pos[1]), int(env.monster_pos[0])] = "M"
            canvas[int(env.blood_pos[1]), int(env.blood_pos[0])] = "B"
            canvas[int(env.agent_pos[1]), int(env.agent_pos[0])] = "A"
            
            # 繪製帶牆壁的地圖
            print("  " + " ".join(["#"] * 12))
            for i, row in enumerate(canvas):
                print("  # " + " ".join(row) + " #")
            print("  " + " ".join(["#"] * 12))
            
            time.sleep(0.05)
            
    agent.update()
    history.append(total_r)

# --- 5. 輸出訓練報表 ---
plt.figure(figsize=(12, 6))
plt.plot(history, color='royalblue', alpha=0.3, label="Raw Episode Reward")
moving_avg = np.convolve(history, np.ones(50)/50, mode='valid')
plt.plot(range(len(moving_avg)), moving_avg, color='crimson', linewidth=2, label="50-Ep Rolling Average")
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title("RPG Agent: Squared Probability (No Dead Neurons) PPO + GAE Training Report")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/squared_ppo_training.png', dpi=150)
plt.show()

print("\n訓練完成！")
print(f"最終 100 輪平均獎勵: {np.mean(history[-100:]):.2f}")
print("使用平方機率分布，無死亡神經元問題！")