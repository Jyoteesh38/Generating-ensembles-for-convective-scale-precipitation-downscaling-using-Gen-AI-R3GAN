import tensorflow as tf

#----------------------------------------------------------------
# Defining loss functions
#----------------------------------------------------------------
MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()
# Relativistic GAN loss (to address mode collapse) with zero-centered gradient penalties (to address non-convergence) - https://openreview.net/pdf?id=VpIH3Wn9eK
# Zero-centered gradient penalties
def Zero_Centered_Gradient_Penalty(samples, x, y_st):
    with tf.GradientTape() as tape:
        tape.watch(samples)
        critics = discriminator([x, y_st, samples], training=True)
        critics_sum = tf.reduce_sum(critics)

    gradient = tape.gradient(critics_sum, samples)
    return tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3])

# Relativistic discriminator loss
def R_Dis_loss(x, y_st, real_samples, fake_samples, real_logits, fake_logits, gamma):

    # R1-penalty - R1 penalizes the gradient norm of D on real data
    R1Penalty = Zero_Centered_Gradient_Penalty(real_samples, x, y_st)
    # R2-penalty - R2 penalizes the gradient norm of D on fake data
    R2Penalty = Zero_Centered_Gradient_Penalty(fake_samples, x, y_st)

    # Calculate Relativistic Logits
    relativistic_logits = real_logits - fake_logits
    # Calculate Adversarial Loss
    adversarial_loss = tf.math.softplus(-relativistic_logits)

    # Discriminator loss
    dis_loss = tf.reduce_mean(adversarial_loss + ((gamma/2)*(R1Penalty + R2Penalty)))

    return dis_loss, tf.reduce_mean(R1Penalty), tf.reduce_mean(R2Penalty)

# Relativistic generator loss
def R_Gen_loss(real_logits, fake_logits):

    # Calculate Relativistic Logits
    relativistic_logits = fake_logits - real_logits
    # Calculate Adversarial Loss
    adversarial_loss = tf.reduce_mean(tf.math.softplus(-relativistic_logits))

    return adversarial_loss
