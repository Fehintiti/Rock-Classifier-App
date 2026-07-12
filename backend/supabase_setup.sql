-- Run this once in the Supabase SQL editor for your project.
-- Creates the table the backend writes user corrections into.

create table if not exists public.feedback (
  id uuid primary key default gen_random_uuid(),
  feedback_id text unique not null,
  image_path text not null,
  model_predicted_type text,
  model_predicted_name text,
  user_corrected_type text,
  user_corrected_name text,
  certainty text,
  created_at timestamptz not null default now()
);

-- Enable RLS with no policies: this denies all access to the anon/authenticated
-- roles by default (neither of which this app uses), while the backend's
-- service_role key continues to work unaffected, since service_role always
-- bypasses RLS regardless of policies.
alter table public.feedback enable row level security;

-- Also create a Storage bucket named "feedback-images" from the Supabase
-- dashboard (Storage -> New bucket). Private is fine — the backend uploads
-- with the service_role key, which has access regardless of bucket visibility.
