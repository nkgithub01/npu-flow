#include <chrono>
#include <thread>
#include <type_traits>

#include <gtest/gtest.h>

#include "engine/sync_timer.hpp"
#include "gtest_helpers.hpp"

using namespace std::chrono_literals;

TEST(SyncTimerBasicFunctionalityTest, TimerDoesNotExpireBeforeElapsedTime) {
  engine::SyncTimer timer(1);
  timer.start();

  EXPECT_FALSE(timer.hasExpired());
}

TEST(SyncTimerBasicFunctionalityTest, TimerExpiresAfterElapsedTime) {
  engine::SyncTimer timer(1);
  timer.start();

  std::this_thread::sleep_for(1100ms);

  EXPECT_TRUE(timer.hasExpired());
}

TEST(SyncTimerBasicFunctionalityTest, TimerWithZeroTimeoutExpiresImmediately) {
  engine::SyncTimer timer(0);
  timer.start();

  EXPECT_TRUE(timer.hasExpired());
}

TEST(SyncTimerBasicFunctionalityTest,
     HasExpiredReturnsFalseBeforeStartIsCalled) {
  engine::SyncTimer timer(1);

  EXPECT_FALSE(timer.hasExpired());
}

TEST(SyncTimerUnlimitedModeTest, UnlimitedTimerNeverExpires) {
  engine::SyncTimer timer(-1);
  timer.start();

  EXPECT_FALSE(timer.hasExpired());

  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(timer.hasExpired());

  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(timer.hasExpired());
}

TEST(SyncTimerUnlimitedModeTest,
     UnlimitedTimerWithOtherNegativeValuesNeverExpires) {
  engine::SyncTimer timer(-100);
  timer.start();

  EXPECT_FALSE(timer.hasExpired());

  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(timer.hasExpired());
}

TEST(SyncTimerUnlimitedModeTest,
     UnlimitedTimerSecondsSinceLastCheckUsesLastCheckTime) {
  engine::SyncTimer timer(-1);
  timer.start();

  std::this_thread::sleep_for(100ms);
  timer.hasExpired();

  std::this_thread::sleep_for(50ms);
  const double elapsed = timer.secondsSinceLastCheck();

  EXPECT_GE(elapsed, 0.04);
  EXPECT_LE(elapsed, 0.15);
}

TEST(SyncTimerRestartTest, RestartResetsTheTimer) {
  engine::SyncTimer timer(1);
  timer.start();

  std::this_thread::sleep_for(500ms);
  EXPECT_FALSE(timer.hasExpired());

  timer.start();

  EXPECT_FALSE(timer.hasExpired());

  std::this_thread::sleep_for(600ms);
  EXPECT_FALSE(timer.hasExpired());

  std::this_thread::sleep_for(500ms);
  EXPECT_TRUE(timer.hasExpired());
}

TEST(SyncTimerElapsedTimeTrackingTest, ElapsedTimeIsZeroBeforeStart) {
  engine::SyncTimer timer(10);

  EXPECT_DOUBLE_EQ(timer.secondsSinceLastCheck(), 0.0);
}

TEST(SyncTimerElapsedTimeTrackingTest,
     ElapsedTimeReturnsTimeSinceStartBeforeFirstHasExpiredCall) {
  engine::SyncTimer timer(10);
  timer.start();

  std::this_thread::sleep_for(100ms);

  const double elapsed = timer.secondsSinceLastCheck();
  EXPECT_GE(elapsed, 0.09);
  EXPECT_LE(elapsed, 0.2);
}

TEST(SyncTimerElapsedTimeTrackingTest,
     ElapsedTimeReturnsTimeSinceLastHasExpiredCall) {
  engine::SyncTimer timer(10);
  timer.start();

  std::this_thread::sleep_for(100ms);
  timer.hasExpired();

  std::this_thread::sleep_for(50ms);

  const double elapsed = timer.secondsSinceLastCheck();
  EXPECT_GE(elapsed, 0.04);
  EXPECT_LE(elapsed, 0.15);
}

TEST(SyncTimerElapsedTimeTrackingTest,
     ElapsedTimeUpdatesCorrectlyAcrossMultipleHasExpiredCalls) {
  engine::SyncTimer timer(10);
  timer.start();

  std::this_thread::sleep_for(100ms);
  timer.hasExpired();

  std::this_thread::sleep_for(50ms);
  const double elapsed1 = timer.secondsSinceLastCheck();

  timer.hasExpired();

  std::this_thread::sleep_for(75ms);
  const double elapsed2 = timer.secondsSinceLastCheck();

  EXPECT_GE(elapsed1, 0.04);
  EXPECT_LE(elapsed1, 0.15);
  EXPECT_GE(elapsed2, 0.06);
  EXPECT_LE(elapsed2, 0.15);
}

TEST(SyncTimerRemainingTimeTest, ReturnsRemainingTimeBeforeExpiration) {
  engine::SyncTimer timer(2);
  timer.start();

  double remaining = timer.secondsRemaining();
  EXPECT_GE(remaining, 1.9);
  EXPECT_LE(remaining, 2.0);

  std::this_thread::sleep_for(500ms);

  remaining = timer.secondsRemaining();
  EXPECT_GE(remaining, 1.4);
  EXPECT_LE(remaining, 1.6);
}

TEST(SyncTimerRemainingTimeTest, ReturnsZeroAfterExpiration) {
  engine::SyncTimer timer(1);
  timer.start();

  std::this_thread::sleep_for(1100ms);

  EXPECT_DOUBLE_EQ(timer.secondsRemaining(), 0.0);
}

TEST(SyncTimerRemainingTimeTest, ReturnsZeroForZeroTimeout) {
  engine::SyncTimer timer(0);
  timer.start();

  EXPECT_DOUBLE_EQ(timer.secondsRemaining(), 0.0);
}

TEST(SyncTimerRemainingTimeTest, ThrowsWhenCalledBeforeStart) {
  engine::SyncTimer timer(10);

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)timer.secondsRemaining(); },
      {"SyncTimer::secondsRemaining() called before start()"});
}

TEST(SyncTimerRemainingTimeTest, ThrowsForUnlimitedTimer) {
  engine::SyncTimer timer(-1);
  timer.start();

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)timer.secondsRemaining(); },
      {"SyncTimer::secondsRemaining() called on unlimited timer"});
}

TEST(SyncTimerLoopUsagePatternTest, TimerWorksCorrectlyInLoopCondition) {
  engine::SyncTimer timer(1);
  timer.start();

  int iterations = 0;
  while (!timer.hasExpired()) {
    ++iterations;
    std::this_thread::sleep_for(100ms);
  }

  EXPECT_GE(iterations, 8);
  EXPECT_LE(iterations, 12);
}

TEST(SyncTimerLoopUsagePatternTest,
     TimerWithLongerTimeoutAllowsMoreIterations) {
  engine::SyncTimer timer(2);
  timer.start();

  int iterations = 0;
  while (!timer.hasExpired()) {
    ++iterations;
    std::this_thread::sleep_for(100ms);
  }

  EXPECT_GE(iterations, 18);
  EXPECT_LE(iterations, 24);
}

TEST(SyncTimerMultipleInstancesTest, MultipleTimersWorkIndependently) {
  engine::SyncTimer timer1(1);
  engine::SyncTimer timer2(2);

  timer1.start();
  timer2.start();

  std::this_thread::sleep_for(1100ms);

  EXPECT_TRUE(timer1.hasExpired());
  EXPECT_FALSE(timer2.hasExpired());

  std::this_thread::sleep_for(1000ms);

  EXPECT_TRUE(timer2.hasExpired());
}

TEST(SyncTimerMoveAndCopySemanticsTest, TimerIsCopyable) {
  EXPECT_TRUE(std::is_copy_constructible_v<engine::SyncTimer>);
  EXPECT_TRUE(std::is_copy_assignable_v<engine::SyncTimer>);
}

TEST(SyncTimerMoveAndCopySemanticsTest, TimerIsMovable) {
  EXPECT_TRUE(std::is_move_constructible_v<engine::SyncTimer>);
  EXPECT_TRUE(std::is_move_assignable_v<engine::SyncTimer>);
}

TEST(SyncTimerTimingAccuracyTest, TimeoutOccursApproximatelyAtSpecifiedTime) {
  engine::SyncTimer timer(1);
  const auto start = std::chrono::steady_clock::now();
  timer.start();

  while (!timer.hasExpired()) {
    std::this_thread::sleep_for(10ms);
  }

  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  EXPECT_GE(elapsed_ms, 950);
  EXPECT_LE(elapsed_ms, 1150);
}
