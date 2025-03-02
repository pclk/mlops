'use client';

import { Paper, Skeleton, Stack, Group } from '@mantine/core';
import { Grid } from '@mantine/core';

export default function SalaryFormSkeleton() {
  return (
    <Paper shadow="xs" p="md" style={{ height: '100%' }}>
      <Stack gap="md">
        {/* Header */}
        <Group justify="space-between" mb="md">
          <Skeleton height={28} width={200} />
          <Skeleton height={24} width={100} />
        </Group>

        {/* Preset Buttons */}
        <Group mb="md">
          <Skeleton height={32} width={120} />
          <Skeleton height={32} width={120} />
          <Skeleton height={32} width={120} />
          <Skeleton height={32} width={80} />
        </Group>

        {/* Form Fields */}
        <Paper withBorder shadow="sm" p="md">
          <Grid>
            <Grid.Col span={6}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Skeleton height={36} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={6}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Skeleton height={36} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={12}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Skeleton height={100} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={6}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Skeleton height={36} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={6}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Skeleton height={36} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={12}>
              <Stack gap="sm">
                <Skeleton height={20} width="50%" />
                <Group>
                  <Skeleton height={36} width={120} />
                  <Skeleton height={36} width={120} />
                  <Skeleton height={36} width={120} />
                </Group>
              </Stack>
            </Grid.Col>
          </Grid>
        </Paper>

        {/* Submit Button */}
        <Skeleton height={36} width={120} />
      </Stack>
    </Paper>
  );
}
