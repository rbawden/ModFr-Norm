#!/usr/bin/perl

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";
use FindBin;
use lib $FindBin::Bin;
use utf8;
use alModFrNormalisation;
use strict;

$| = 1;

my $lefff_mlex_file;
my $apply_nontrivial_regexps;

while (1) {
  $_ = shift;
  if (/^$/) {last}
  elsif (/^-r$/) {$apply_nontrivial_regexps = 1}
  elsif (/^-l$/) {$lefff_mlex_file = shift || die "ERROR: option '-l' must be followed by the location of the Lefff mlex file";}
  else {die "Unknown option '$_'"}
}

my (%lefff,%mapping_to_lefff);

if ($lefff_mlex_file ne "") {
  die "ERROR: Lefff mlex file '$lefff_mlex_file' could not be found" unless -r $lefff_mlex_file;
  open MLEX, "<$lefff_mlex_file" || die "ERROR: Lefff mlex file '$lefff_mlex_file' could not be opened";
  binmode MLEX, ":utf8";
  my $mod;
  while (<MLEX>) {
    chomp;
    die unless /^([^\t]+)\t[^\t]+\t[^\t]+(?:\t[^\t]*)$/;
    $lefff{$1} = 1;
  }
  close MLEX;
  for my $w (keys %lefff) {
    for my $mod (vowelcircumflex_to_vowel_s($w),eacute_to_e_s($w),remove_diacritics($w),vowel_u_to_vowel_v($w),consonant_v_to_consonant_u($w),y_to_i($w)) {
      if (!defined($lefff{$mod})) {
	if (defined($mapping_to_lefff{$mod})) {
	  $mapping_to_lefff{$mod} = "__AMBIGUOUS__";
	} else {
	  $mapping_to_lefff{$mod} = $w;
	}
      }
    }
  }
}

while (<>) {
  chomp;
  $_ = almodfrnormalise($_,$apply_nontrivial_regexps,\%mapping_to_lefff);
  print $_."\n";
}

sub remove_diacritics {
  my ($s) = @_;
  $s =~ tr/ǽǣáàâäąãăåćčçďéèêëęěğìíîĩĭıïĺľłńñňòóôõöøŕřśšşťţùúûũüǔỳýŷÿźẑżžÁÀÂÄĄÃĂÅĆČÇĎÉÈÊËĘĚĞÌÍÎĨĬİÏĹĽŁŃÑŇÒÓÔÕÖØŔŘŚŠŞŤŢÙÚÛŨÜǓỲÝŶŸŹẐŻŽ/ææaaaaaaaacccdeeeeeegiiiiiiilllnnnoooooorrsssttuuuuuuyyyyzzzzAAAAAAAACCCDEEEEEEGIIIIIIILLLNNNOOOOOORRSSSTTUUUUUUYYYYZZZZ/;
  $s =~ s/œ/oe/g;
  $s =~ s/æ/ae/g;
  $s =~ s/ƣ/oi/g;
  $s =~ s/ĳ/ij/g;
  $s =~ s/ȣ/ou/g;
  $s =~ s/Œ/OE/g;
  $s =~ s/Æ/AE/g;
  $s =~ s/Ƣ/OI/g;
  $s =~ s/Ĳ/IJ/g;
  $s =~ s/Ȣ/OU/g;
  return $s;
}

sub vowel_u_to_vowel_v {
  my ($s) = @_;
  $s =~ s/([aeiou])u/$1v/g;
  return $s;
}

sub consonant_v_to_consonant_u {
  my ($s) = @_;
  $s =~ s/([^aeiou])v/$1u/g;
  return $s;
}

sub y_to_i {
  my ($s) = @_;
  $s =~ tr/y/i/;
  return $s;
}

sub eacute_to_e_s {
  my ($s) = @_;
  $s =~ s/é/es/g;
  return $s;
}

sub vowelcircumflex_to_vowel_s {
  my ($s) = @_;
  $s =~ s/â/as/g;
  $s =~ s/ê/es/g;
  $s =~ s/î/is/g;
  $s =~ s/ô/os/g;
  $s =~ s/û/us/g;
  return $s;
}
