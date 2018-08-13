package cn.sibat.ksh

object SelectT {

  val n_states = 6 //多少种状态
  val action = Array("left", "right") //行为
  val epsilon = 0.9 //选择行为的概率，0.9代表90%的概率选择学习后的结果，10%的概率选择随机行为
  val learning_rate = 0.1 // 学习率
  val gama = 0.9 //状态衰减率
  val max_learn = 13 //学习次数

  /**
    * 建立Q-table
    *
    * @param n_states 状态数
    * @param action   动作
    * @return q-table
    */
  def build_q_table(n_states: Int, action: Array[String]): Array[Array[Double]] = {
    Array()
  }

  /**
    * 根据当前的状态和q-table，选择执行的行为
    *
    * @param state   当前状态
    * @param q_table q-table
    * @return 选择的行为
    */
  def choose_action(state: Int, q_table: Array[Array[Double]]): String = {
    //1.当状态对应的行为的概率
    //2. 选随机数 > epsilon 或者 处于初始化状态，随机选取行为
    //3. 非2的情况下，选取概率最大的行为
    ""
  }

  /**
    * 采取行为后，环境的反馈
    *
    * @param S 状态
    * @param A 行为
    * @return (下一状态，奖励)
    */
  def get_env_feedback(S: Int, A: String): (Int, Int) = {
    var S_ = 0
    var R = 0
    //1.行为右移，终点的前一位置，右移得分，下一状态为终止，正向奖励，否则，下一状态右移，不做奖励
    //2. 行为左移，起点位置，左移还为起点，其他状态左移，不做奖励
    (S_, R)
  }

  /**
    * 更新环境
    *
    * @param S            单前状态
    * @param episode      学习次数
    * @param step_counter 所需成本
    */
  def update_env(S: Int, episode: Int, step_counter: Int): Unit = {
    val line = "-," * (n_states - 1)
    val env_list = line.split(",") ++ Array("T")
    if (S == -1) {
      print(s"\rEpisode ${episode + 1}: total_steps = $step_counter")
      Thread.sleep(1000)
      print("\r")
    } else {
      env_list(S) = "o"
      print("\r" + env_list.mkString(""))
      Thread.sleep(300)
    }
  }

  /**
    * 学习程序
    *
    * @return
    */
  def learn(): Array[Array[Double]] = {
    //1. 构建Q-table
    val q_table = build_q_table(n_states, action)
    //2. repeat
    //3. loop:当前状态选择行为
    //4. loop:当前状态和选择行为在环境中的反馈
    //5.loop: 理论采取当前状态采取当前的行为的反馈q值
    //6. loop:实际上反馈的q值
    //7. loop:更新q-table
    q_table
  }

  def main(args: Array[String]): Unit = {
    val qtable = learn()
    qtable.foreach(s => println(s.mkString(",")))
  }
}
