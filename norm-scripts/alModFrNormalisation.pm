package alModFrNormalisation;
use utf8;
use strict;
use Exporter;
our @ISA = 'Exporter';
our @EXPORT = qw(&almodfrnormalise);

sub almodfrnormalise {
  my ($apply_nontrivial_regexps,$mapping_to_lefff);
  ($_,$apply_nontrivial_regexps,$mapping_to_lefff) = @_;
  s/ //g;
  tr/ſ/s/;
  s/ß/ss/g;
  s/&/et/g;
  s/ẽ([mbp])/em$1/g;
  s/ẽ/en/g;
  s/ã([mbp])/am$1/g;
  s/ã/an/g;
  s/õ([mbp])/om$1/g;
  s/õ/on/g;
  s/ũ([mbp])/um$1/g;
  s/ũ/un/g;
  if ($apply_nontrivial_regexps) {
    s/\bI([aeE])/J$1/g;
    s/\bi([aeou])/j$1/g;
    s/([uaoncü])y\b/$1i/g;
    s/\b((?:ce)?)([lh])i\b/$1$2ui/g;
    s/\b([mRft])i\b/$1oi/g;
    s/\bvi\b/vois/g;
    s/\b(vr)i\b/$1ai/g;
    s/\bvu([aeiou])/v$1/g;
    s/\bvn/un/g;
    s/\bI'/J'/g;
    s/\bi'/j'/g;
    s/\bi\b/ai/g;
    s/es(tr|m)e((?:s|nt)?)\b/ê$1e$2/g;
    s/as(tr|m)e((?:s|nt)?)\b/â$1e$2/g;
    s/is(tr|m)e((?:s|nt)?)\b/î$1e$2/g;
    s/us(tr|m)e((?:s|nt)?)\b/û$1e$2/g;
    s/os(tr|m)e((?:s|nt)?)\b/ô$1e$2/g;
    s/s(tr|m)e((?:s|nt)?)\b/$1e$2/g;
    s/é(mes?)\b/ê$1/g;
    s/au([eo])/av$1/g;
    s/\besté\b/été/g;
    s/\bestes\b/êtes/g;
    s/estoi((?:en)?)t\b/étai$1t/g;
    s/(ser|aur|av|fais)oi((?:en)?)t\b/$1ai$2t/g;
    s/^A\b/À/;
    s/\b(ap|t)res/$1rès/g;
    s/\ba(mes?)\b/â$1/g;
    s/\bii\b/ici/g;
    s/sçi\b/sais/g;
    s/sç/s/g;
    s/([^aeou])e(res?)\b/$1è$2/g;
    s/to[ûu](?:si|j)ours?/toujours/g;
    s/([aeiou])i([aiou])/$1j$2/g;
    s/([aeiou])u([aeiou])/$1v$2/g;
    s/usm/ûm/g;
    s/osm/ôm/g;
    s/ë\b/e/g;
    s/avecques?/avec/g;
    s/aprés\b/après/g;
    s/\bes(tan?t)/é$1/g;
  }
  my $out;
  if (defined($mapping_to_lefff)) {
    for my $w (split /\b/, $_) {
      if ($w =~ / / || $w =~ /^.$/ || $w =~ /^[A-Z]/ || !defined($mapping_to_lefff->{$w}) || $mapping_to_lefff->{$w} eq "__AMBIGUOUS__") {
	$out .= $w;
      } else {
	$out .= $mapping_to_lefff->{$w};
	#      print STDERR "$w>".$mapping_to_lefff->{$w}."\n";
      }
    }
    return $out;
  } else {
    return $_;
  }
}
1;
